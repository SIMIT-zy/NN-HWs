import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# 生成器结构
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, mask):
        residual = self.relu(self.conv1(x))
        residual = self.conv2(residual)
        return x + mask * residual  # 仅缺失区域更新

class Generator(nn.Module):
    def __init__(self, n_blocks=2):
        super().__init__()
        self.encoder = nn.Conv2d(opt.channels, 32, 3, padding=1)
        self.res_blocks = nn.ModuleList([ResBlock(32) for _ in range(n_blocks)])
        self.decoder = nn.Conv2d(32, opt.channels, 3, padding=1)

    def forward(self, img, mask):
        x = self.encoder(img)
        for block in self.res_blocks:
            x = block(x, mask)
        x = self.decoder(x)
        return (1.0 - mask.float()) * img + mask * x  # 合并输出


# 鉴别器结构
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Initialize weights
generator.apply(weights_init_normal)

# 根据cuda变量创建设备对象
device = torch.device("cuda" if cuda else "cpu")
# 2. 加载权重文件
discriminator_weights = torch.load("./models/discriminator_DCGAN_gray.pth", map_location=device)
# 3. 应用权重
discriminator.load_state_dict(discriminator_weights)
discriminator.eval()

# 3. 加载预训练模型（保持不变）
from pytorchcv.model_provider import get_model
net = get_model("resnet56_cifar10", pretrained=False)
state_dict = torch.load("./models/resnet56_cifar10-0452-628c42a2.pth")
net.load_state_dict(state_dict)
net.eval()
loss = nn.CrossEntropyLoss(reduction='none')

if cuda:
    adversarial_loss.cuda()
    generator.cuda()
    discriminator.cuda()
    net.cuda()


# Configure data loader
from torch.utils.data import DataLoader, Dataset
import os
import pickle
from PIL import Image
class AdvCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        with open(os.path.join(root, 'train' if train else 'test'), 'rb') as f:
            entry = pickle.load(f)
            self.data = entry['data']
            self.labels = entry['labels']

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[index]

    def __len__(self):
        return len(self.data)

dataloader = torch.utils.data.DataLoader(
    AdvCIFAR10(
        root='./datas/adv_cifar10',
        train=True,
        transform=transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
        ])
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image(dataloader, batches_done=None):
    """
    从 dataloader 中提取10张不同类别的图像，生成对应输出，并用 Matplotlib 左右对比显示
    排列为 10x2 网格（10行2列），每行：真实图像 | 生成图像
    """
    # 1. 从 dataloader 中收集10张不同类别的图像
    real_images = []
    real_labels = []
    used_labels = set()

    for images, labels in dataloader:
        for img, lbl in zip(images, labels):
            if lbl.item() not in used_labels:
                real_images.append(img)
                real_labels.append(lbl)
                used_labels.add(lbl.item())
                if len(used_labels) == 10:
                    break
        if len(used_labels) == 10:
            break

    # 转换为Tensor
    real_images = torch.stack(real_images).to(device)  # [10, C, H, W]
    real_labels = torch.stack(real_labels).to(device)  # [10]

    # 2. 使用真实图像和原标签生成图像
    with torch.no_grad():
        mask = (real_images == 0)
        gen_images = generator(real_images, mask)  # 假设generator支持图像输入
        #gen_images = gen_images.clamp(real_images.min(), real_images.max())
        gen_images = gen_images.clamp(0, 1)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # 3. 使用 Matplotlib 创建对比图
    fig, axes = plt.subplots(10, 2, figsize=(6, 20))  # 10行2列
    fig.subplots_adjust(hspace=0.4, wspace=0.1)  # 调整间距

    for i in range(10):
        # 真实图像 (左)
        real_img = real_images[i].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        axes[i, 0].imshow(real_img, cmap='gray')
        axes[i, 0].set_title(f"Real (Class {classes[real_labels[i].item()]})", fontsize=8)
        axes[i, 0].axis('off')

        # 生成图像 (右)
        gen_img = gen_images[i].permute(1, 2, 0).cpu().numpy()
        axes[i, 1].imshow(gen_img, cmap='gray')
        axes[i, 1].set_title(f"Generated", fontsize=8)
        axes[i, 1].axis('off')

    # 4. 保存图像
    plt.savefig(f"./images/step3/results_200/comparison_{batches_done}.png", bbox_inches='tight', dpi=150)
    plt.close()  # 关闭图形，避免内存泄漏

def optimize_display(output_history, classes):
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.plot(output_history[:, i], label=f'{classes[i]}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses Trends Over Epochs')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./images/step3/optimize_display.png")
    # plt.show()

# ----------
#  Training
# ----------
loss_history = torch.zeros((opt.n_epochs, 3))
mean_tensor = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
std_tensor = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

for epoch in tqdm(range(opt.n_epochs)):
    for imgs, labels in dataloader:

        # Adversarial ground truths
        valid = Variable(FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        z = Variable(imgs.type(FloatTensor))
        gen_labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        mask = (z == 0)
        gen_imgs = generator(z, mask)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator((gen_imgs - 0.5) / 0.5)
        d_loss = adversarial_loss(validity, valid)

        y_hat = net((gen_imgs - std_tensor) / mean_tensor)
        c_loss = loss(y_hat, gen_labels).mean()

        g_loss = 0.1 * c_loss + 0.9 * d_loss + c_loss * d_loss
        g_loss.backward()
        optimizer_G.step()

    loss_condition = torch.tensor([c_loss.item(), d_loss.item(), g_loss.item()])
    loss_history[epoch] = loss_condition
    print(epoch, loss_condition)
    if (epoch+1) % 10 == 0:
        sample_image(dataloader, batches_done=epoch+1)

    if epoch == 1:
        sample_image(dataloader, batches_done=epoch+1)
        optimize_display(loss_history[:epoch+1, :], ["c_loss", "d_loss", "g_loss"])
        torch.save(generator.state_dict(), './models/generator_repair.pth')

optimize_display(loss_history[:epoch+1, :], ["c_loss", "d_loss", "g_loss"])
torch.save(generator.state_dict(), './models/generator_repair.pth')