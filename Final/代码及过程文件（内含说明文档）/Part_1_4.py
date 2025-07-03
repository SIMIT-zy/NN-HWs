import argparse
# import numpy as np
import torch
import torchvision.transforms as transforms
# from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch import nn
from torchmetrics.image import FrechetInceptionDistance, InceptionScore
from tqdm import tqdm


# 1. 参数配置
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    # parser.add_argument("--lr_g", type=float, default=0.0002)
    # parser.add_argument("--lr_d", type=float, default=0.0002)
    # parser.add_argument("--b1", type=float, default=0.5)
    # parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    return parser.parse_args()

# 2. 数据预加载（避免重复IO）
def load_real_data(n_eval_samples, device):
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.CIFAR10(root='./datas', train=True, download=False, transform=transform)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    # 预分配张量（避免append）
    real_data = torch.zeros(
        (n_eval_samples, opt.channels, opt.img_size, opt.img_size),
        dtype=torch.uint8,
        device=device
    )

    # 直接填充数据
    idx = 0
    for batch, _ in loader:
        batch = (batch * 127.5 + 127.5).clamp(0, 255).type(torch.uint8).to(device)
        batch_size = batch.size(0)

        # 确保不越界
        if idx + batch_size > n_eval_samples:
            batch = batch[:n_eval_samples - idx]
            batch_size = batch.size(0)

        real_data[idx: idx + batch_size] = batch
        idx += batch_size

        if idx >= n_eval_samples:
            break

    return real_data[:idx]  # 返回实际填充的数据


# 3. 生成器定义
# 条件批归一化层 (核心修改)
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, cond_size):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Linear(cond_size, num_features)
        self.beta = nn.Linear(cond_size, num_features)

    def forward(self, x, cond):
        normalized = self.bn(x)
        gamma = self.gamma(cond).view(x.size(0), -1, 1, 1)
        beta = self.beta(cond).view(x.size(0), -1, 1, 1)
        return gamma * normalized + beta

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4
        # 条件投影层
        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2)
        )

        # 初始特征图生成
        self.l2 = nn.Linear(512, 128 * self.init_size ** 2)

        # 上采样块 (使用条件批归一化)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.cbn1 = ConditionalBatchNorm2d(128, 512)

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.cbn2 = ConditionalBatchNorm2d(64, 512)

        self.final = nn.Sequential(
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_labels = self.l1(self.label_emb(labels))  # [batch, 512]
        gen_noise = self.l1(noise)  # [batch, 512]
        out = 0.5 * gen_labels + 0.5 * gen_noise

        # 初始特征图
        img = self.l2(out)
        img = img.view(img.shape[0], 128, self.init_size, self.init_size)  # [batch, 128, 4, 4]

        # 上采样 + 条件注入 + 噪声注入
        img = self.block1(img)  #[batch, 128, 8, 8]
        img = self.cbn1(img, out)
        img = img + 0.01 * torch.randn_like(img)  # 添加层间噪声

        img = self.block2(img)  #[batch, 64, 16, 16]
        img = self.cbn2(img, out)
        img = img + 0.05 * torch.randn_like(img)  # 添加层间噪声

        img = self.final(img)  #[batch, 3, 32, 32]
        return img


# 4. 评估函数（复用Inception模型）
def evaluate_gan(generator, real_data, device):
    # 初始化指标（复用模型）
    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception = InceptionScore().to(device)

    # 评估真实数据
    fid.update(real_data, real=True)

    # 评估生成数据
    generator.eval()
    with torch.no_grad():
        for _ in range(0, n_eval_samples, opt.batch_size):
            z = torch.randn(opt.batch_size, opt.latent_dim).to(device)
            labels = torch.randint(0, opt.n_classes, (opt.batch_size,)).to(device)
            gen_images = generator(z, labels)

            # 调整尺寸并归一化到[0,255]
            gen_images = transforms.functional.resize(gen_images, [299, 299], antialias=True)
            gen_images = (gen_images * 127.5 + 127.5).clamp(0, 255).type(torch.uint8)

            # 更新指标
            fid.update(gen_images, real=False)
            inception.update(gen_images)

            # 显存清理
            del z, labels, gen_images
            torch.cuda.empty_cache()

    return fid.compute(), inception.compute()


# 5. 主流程
# 定义参数
opt = parse_args()
n_eval_samples = 10000
n_runs = 10

# 根据cuda变量创建设备对象
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

# 加载数据集
real_data = load_real_data(n_eval_samples, device=device)
# 加载生成器
generator = Generator().to(device)
generator.load_state_dict(torch.load("./models/generator_opt2.pth", map_location=device))

# 多次评估取平均
# 预分配PyTorch张量（自动适配GPU/CPU）
fid_scores = torch.zeros(n_runs, device=device)
is_means = torch.zeros(n_runs, device=device)
is_stds = torch.zeros(n_runs, device=device)

for i in tqdm(range(n_runs)):
    fid, (is_mean, is_std) = evaluate_gan(generator, real_data, device)
    fid_scores[i] = fid
    is_means[i] = is_mean
    is_stds[i] = is_std
    print(f"Run {i+1}: FID={fid:.2f}, IS={is_mean:.2f}±{is_std:.2f}")

# 输出最终结果
print(f"FID: {torch.mean(fid_scores):.2f} ± {torch.std(fid_scores):.2f}")
print(f"IS: {torch.mean(is_means):.2f} ± {torch.mean(is_stds):.2f}")