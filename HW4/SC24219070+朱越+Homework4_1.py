import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from pytorchcv.model_provider import get_model
from tqdm import *

#预训练的网络参数使用了标准化的数据集，因此在预处理中引入标准化
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]
transform = transforms.Compose([
    transforms.ToTensor(),  # 自动缩放到 [0,1]
    transforms.Normalize(mean, std)  #执行标准化
])
#加载数据集，由于是预先下载好的，所以download使用False
testset = CIFAR10(root='datas', train=False, download=False, transform=transform)

#加载网络，网络权重也都下载好了
#resnet56_cifar10预训练网络的Test Accuracy测试结果为95.32%
net = get_model("resnet56_cifar10", pretrained=False)
state_dict = torch.load("./models/resnet56_cifar10-0452-628c42a2.pth")
net.load_state_dict(state_dict)
net.eval()

# 主要参数
lr = 0.01  # 学习率
max_iter = 100  # 最大迭代次数
c = 1  # 平衡松弛函数的超参数
kappa = 1  # 松弛函数中的置信度参数
epsilon = 8 / 255  # L无穷范数限制

correct = 0  #识别准确个数
success = 0  #攻击成功个数（定向）

#依次攻击，所以batch_size为1
test_loader = DataLoader(testset, 1, shuffle=False)
for data, label in tqdm(test_loader):
    #在攻击之前先检查一下是否正确识别
    output = net(data)
    init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    # If the initial prediction is wrong, don't bother attacking, just move on
    if init_pred.item() != label.item():
        continue

    #设置一个目标，这里得到的是概率值次大的索引
    target = torch.max(output[:, [l for l in range(10) if l != label]], dim=1)[1]
    # 初始化扰动变量（需梯度跟踪）
    delta = torch.zeros_like(data, requires_grad=True)
    optimizer = torch.optim.SGD([delta], lr=lr)

    for i in range(max_iter):
        optimizer.zero_grad()
        # 生成对抗样本
        data_denorm = data * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)
        perturbed_data = data_denorm + delta
        adv_image = (perturbed_data - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
        # 模型预测
        logits = net(adv_image)
        # 计算松弛函数
        logits_target = logits[:, target]
        logits_other = torch.max(logits[:, [i for i in range(logits.shape[1]) if i != target]], dim=1)[0]
        f6 = torch.clamp(logits_other - logits_target + kappa, min=0)  #PPT写的f6，所以这里也用f6
        # 优化目标
        loss = c * f6
        #早停减少迭代次数
        #在这里早停使用的是上一轮优化的delta，经过了上一轮的clamp操作
        if loss <= 1e-4:
            break
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        # 投影操作：将扰动裁剪到[-epsilon, epsilon]范围内
        delta.data.clamp_(-epsilon, epsilon)

    # 最终对抗样本
    data_denorm = data * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)
    perturbed_data = data_denorm + delta.detach()  #抛弃梯度信息
    adv_image = (perturbed_data - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    # 计算正确率
    adv_logits = net(adv_image)
    adv_pred = adv_logits.max(1, keepdim=True)[1]
    if adv_pred.item() == label.item():  #识别正确
        correct += 1
    if adv_pred.item() == target.item():  #攻击成功
        success += 1

# Calculate final accuracy for this epsilon
final_acc = correct / float(len(test_loader))
final_suc = success / float(len(test_loader))
print(f"Test Accuracy = {final_acc}, Attack Success Rate = {final_suc}")
