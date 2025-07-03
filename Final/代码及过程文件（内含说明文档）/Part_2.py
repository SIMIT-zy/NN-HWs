import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from pytorchcv.model_provider import get_model
from PIL import Image
from tqdm import tqdm
import pickle

# 1. 初始化配置（保持不变）
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 2. 加载原始数据集（保持不变）
testset = CIFAR10(root='datas', train=False, download=False, transform=transform)

# 3. 加载预训练模型（保持不变）
net = get_model("resnet56_cifar10", pretrained=False)
state_dict = torch.load("./models/resnet56_cifar10-0452-628c42a2.pth")
net.load_state_dict(state_dict)
net.eval()


# 4. 优化后的Damage函数（保持SGD和原始逻辑）
def Damage(data, target):
    # 初始化（保持原参数）
    lr = 0.01
    dr = 0.9
    max_iter = 200
    delta = torch.rand_like(data, requires_grad=True)
    optimizer = torch.optim.SGD([delta], lr=lr)

    # 预计算标准化张量（加速但数学等价）
    mean_tensor = torch.tensor(mean, device=data.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=data.device).view(1, 3, 1, 1)

    for epoch in range(max_iter):
        optimizer.zero_grad()

        # 原始掩膜生成逻辑（向量化实现）
        figure = delta.sum(dim=1, keepdim=True)
        k = int(dr * figure.numel())
        kth_value = torch.kthvalue(figure.flatten(), k).values
        mask = (figure >= kth_value).float().expand(-1, 3, -1, -1)

        # 原始对抗样本生成流程
        perturbed_data = data * std_tensor + mean_tensor
        adv_image = (perturbed_data * mask - mean_tensor) / std_tensor

        # 原始损失计算
        logits = net(adv_image)
        logits_target = logits[:, target]
        logits_other = logits.abs()[:, [i for i in range(10) if i != target]].sum(1)
        loss = logits_other - logits_target

        # 原始早停条件
        if dr < 0.6 and logits_target > (logits_other + 1):
            break

        loss.backward()
        optimizer.step()
        dr *= (1 - lr)  # 保持原始衰减

    return adv_image.detach(), logits.argmax(1)


# 5. 优化后的create_adv_dataset（预分配内存+保持格式）
def create_adv_dataset(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    test_loader = DataLoader(testset, 1, shuffle=False)

    # 预分配内存（保持CIFAR10的NHWC uint8格式）
    max_samples = len(testset)
    adv_data = np.zeros((max_samples, 32, 32, 3), dtype=np.uint8)
    adv_labels = np.zeros(max_samples, dtype=np.int64)
    valid_count = 0

    for data, target in tqdm(test_loader, desc="Generating Adversarial Examples"):
        # 初始预测检查（保持不变）
        with torch.no_grad():
            output = net(data)
            init_pred = output.argmax(1)
            if init_pred.item() != target.item():
                continue

        # 生成对抗样本
        image, label = Damage(data, target)

        # 验证对抗样本（保持不变）
        with torch.no_grad():
            output = net(image)
            final_pred = output.argmax(1)
            if final_pred.item() != label.item():
                continue

        # 反标准化并存储（直接写入预分配数组）
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # CHW->HWC
        image = np.clip((image * std + mean) * 255, 0, 255).astype(np.uint8)
        adv_data[valid_count] = image
        adv_labels[valid_count] = label.item()
        valid_count += 1

    # 裁剪到实际有效样本
    adv_data = adv_data[:valid_count]
    adv_labels = adv_labels[:valid_count]

    # 保持原始划分逻辑
    indices = np.random.permutation(valid_count)
    split = int(0.9 * valid_count)

    # 保持原始pickle保存结构
    with open(os.path.join(output_dir, 'train'), 'wb') as f:
        pickle.dump({
            'data': adv_data[indices[:split]],
            'labels': adv_labels[indices[:split]]
        }, f)

    with open(os.path.join(output_dir, 'test'), 'wb') as f:
        pickle.dump({
            'data': adv_data[indices[split:]],
            'labels': adv_labels[indices[split:]]
        }, f)

    # 保持原始meta信息
    with open(os.path.join(output_dir, 'meta'), 'wb') as f:
        pickle.dump({
            'num_cases_per_batch': 10000,
            'label_names': testset.classes
        }, f)

    print(f"Generated {valid_count} adversarial samples (Train: {split}, Test: {valid_count - split})")


# 6. 保持原始数据集类（完全不变）
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


# 7. 执行生成（保持不变）
create_adv_dataset(output_dir='./datas/adv_cifar10')
print("Done!")