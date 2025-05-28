import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from pytorchcv.model_provider import get_model
from torch.nn import NLLLoss
from tqdm import *

#加载网络参数
def ptcv_get_model(name, model_path):
    net = get_model(name, pretrained=False)
    state_dict = torch.load(model_path)
    net.load_state_dict(state_dict)
    return net

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    #这里的归一化替换为了标准化，在攻击外部
    # Return the perturbed image
    return perturbed_image

#预训练的网络参数使用了标准化的数据集，因此在预处理中引入标准化
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]
transform = transforms.Compose([
    transforms.ToTensor(),  # 自动缩放到 [0,1]
    transforms.Normalize(mean, std)  #执行标准化
])
#加载数据集，由于是预先下载好的，所以download使用False
testset = CIFAR10(root='datas', train=False, download=False, transform=transform)

#加载网络，网络权重也都下载好了，使用了自定义的加载函数
black_box_net = ptcv_get_model("resnet56_cifar10",
                               "./models/resnet56_cifar10-0452-628c42a2.pth")
shadow_net = ptcv_get_model("resnet20_cifar10",
                            "./models/resnet20_cifar10-0597-9b0024ac.pth")
# Set the model in evaluation mode. In this case this is for the Dropout layers
black_box_net.eval()
shadow_net.eval()

#设置参数
epsilon = 8/255  #L无穷范数限制
# Accuracy counter
correct = 0

#依次攻击，所以batch_size为1
test_loader = DataLoader(testset, 1, shuffle=False)
# Loop over all examples in test set
for data, target in tqdm(test_loader):
    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    #利用影子网络识别并提取梯度
    # Forward pass the data through the model
    output = shadow_net(data)
    #由于是黑盒攻击，所以不排除识别错误的项目了
    # Calculate the loss
    criterion = NLLLoss()
    loss = criterion(output, target)
    # Zero all existing gradients
    shadow_net.zero_grad()
    # Calculate gradients of model in backward pass
    loss.backward()
    # Collect ``datagrad``
    data_grad = data.grad.data
    #这里的反标准化改写了，没有使用自定义函数
    data_denorm = data * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)
    # Call FGSM Attack
    perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)
    #再执行标准化
    turbed_image = (perturbed_data - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)

    #扰动后的图像用于攻击黑盒网络
    # Re-classify the perturbed image
    output = black_box_net(turbed_image)
    # Check for success
    final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    if final_pred.item() == target.item():
        correct += 1

# Calculate final accuracy for this epsilon
final_acc = correct/float(len(test_loader))
print(f"Test Accuracy = {correct} / {len(test_loader)} = {final_acc}")