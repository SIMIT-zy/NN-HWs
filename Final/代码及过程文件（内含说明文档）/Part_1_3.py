import argparse
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

# from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
# import torch.nn.functional as F
import torch

from tqdm import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr_g", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--lr_d", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
# parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


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
        out = 0.45 * gen_labels + 0.55 * gen_noise

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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True, dp=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            if dp:
                block.append(nn.Dropout2d(0.25))
            return block

        # 图像特征提取
        self.img_encoder = nn.Sequential(
            # 输入: (3, 32, 32)
            *discriminator_block(opt.channels, 16, bn=False), # 输出: (16, 16, 16)
            *discriminator_block(16, 32),  # 输出: (32, 8, 8)
            *discriminator_block(32, 64),  # 输出: (64, 4, 4)
            *discriminator_block(64, 128, dp=False),  # 输出: (128, 2, 2)

            nn.Flatten(),  # 输出: (batch, 128*2*2) = (batch, 512)
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2)
        )

        # 标签处理路径
        self.label_embedding = nn.Sequential(
            nn.Embedding(opt.n_classes, 512),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 联合鉴别器 (维度对齐)
        self.model = nn.Sequential(
            nn.Linear(1536, 512),  # 512(img) + 512(label) +512(cross) = 1536
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # 处理图像
        img_in = self.img_encoder(img)  # [batch, 512]
        # 处理类别
        label_in = self.label_embedding(labels)  # [batch, 512]

        # 特征融合 (维度对齐)
        d_in = torch.cat([img_in, label_in, img_in * label_in], dim=1)  # [batch, 1536]
        validity = self.model(d_in)
        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

'''
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
'''

# 根据cuda变量创建设备对象
device = torch.device("cuda" if cuda else "cpu")

# 2. 加载权重文件
generator_weights = torch.load("./models/generator_opt1.pth", map_location=device)
discriminator_weights = torch.load("./models/discriminator_opt1.pth", map_location=device)

# 3. 应用权重
generator.load_state_dict(generator_weights)
discriminator.load_state_dict(discriminator_weights)

# Configure data loader
#加载数据集，由于是预先下载好的，所以download使用False
dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root='./datas',
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
# 使用了不同的lr
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    # 做好归一化
    image = (gen_imgs.data.clamp_(-1, 1) + 1) / 2
    save_image(image, "./images/step1/results_opt2/%d.png" % batches_done, nrow=n_row, normalize=True)

def optimize_display(output_history, classes):
    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.plot(output_history[:, i], label=f'{classes[i]}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses Trends Over Epochs')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./images/step1/optimize_opt2_display.png")
    # plt.show()

# ----------
#  Training
# ----------

loss_history = torch.zeros((opt.n_epochs, 4))

for epoch in tqdm(range(opt.n_epochs)):
    for i, (imgs, labels) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, imgs.shape[0])))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid - 0.1)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake + 0.1)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    loss_condition = torch.tensor([g_loss.item(), d_real_loss.item(), d_fake_loss.item(), d_loss.item()])
    loss_history[epoch] = loss_condition
    print(epoch, loss_condition)

    if (epoch >= 20
            and torch.abs(torch.mean(loss_history[epoch-10:epoch, 0])) - 0.693 < 0.1
            and torch.abs(torch.mean(loss_history[epoch-10:epoch, 3])) - 0.693 < 0.1
    ):
        sample_image(n_row=10, batches_done=epoch+1)
        break

    if (epoch+1) % 10 == 0:
        sample_image(n_row=10, batches_done=epoch+1)

optimize_display(loss_history[:epoch+1, :], ["g_loss", "d_real_loss", "d_fake_loss", "d_loss"])
torch.save(generator.state_dict(), './models/generator_opt2.pth')
torch.save(discriminator.state_dict(), './models/discriminator_opt2.pth')