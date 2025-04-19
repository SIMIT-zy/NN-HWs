import torch
import pandas
import random
import matplotlib.pyplot as plt

#①准备数据
data_file = r'datas/HW1.csv'
#end①

data = pandas.read_csv(data_file)
data = torch.tensor(data.to_numpy(), dtype=torch.float32)

# 计算相关系数矩阵
correlation_matrix = torch.corrcoef(data.T)
# 提取 features 和 labels 之间的相关性
correlations = correlation_matrix[:-1, -1]

inputs = torch.tensor([])
for num in range(correlations.shape[0]):
    if torch.abs(correlations[num]) >= 0.4:
        inputs = torch.cat((inputs, data[:, num].unsqueeze(1)), dim=1)
#inputs = data[:, 0:-1].clone()
outputs = data[:, -1].clone()
features = inputs.clone()
labels = outputs.clone()

def features_scaling(features, f_means, f_stds):
    for num in range(features.shape[1]):
        mean = torch.mean(features[:, num])
        std = torch.std(features[:, num])
        f_means[num] = mean
        f_stds[num] = std
        features[:, num] = (features[:, num] - mean) / std

features_means = torch.zeros(features.shape[1])
features_stds = torch.ones(features.shape[1])
features_scaling(features, features_means, features_stds)

def data_iter(batch_size, features, labels):
    num_examples = features.shape[0]
    indices = torch.arange(0, num_examples)
    random.shuffle(indices)
    for num in range(0, num_examples, batch_size):
        batch_indices = indices[num: min(num + batch_size, num_examples)].clone().detach()
        yield features[batch_indices], labels[batch_indices]

def linreg(x, w, b):
    """线性回归模型"""
    return torch.matmul(x, w) + b

def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2

def Adam(params, lr, Momentum_terms, RMSProp_terms, beta_Momentum, beta_RMSProp, t, batch_size, eps):
    """Adam方法"""
    t += 1
    with torch.no_grad():
        for num, (param, Momentum_term, RMSProp_term) in enumerate(zip(params, Momentum_terms, RMSProp_terms)):
            RMSProp_terms[num] = beta_RMSProp * RMSProp_term + (1 - beta_RMSProp) * (param.grad / batch_size) ** 2
            Momentum_terms[num] = beta_Momentum * Momentum_term + (1 - beta_Momentum) * param.grad / batch_size
            Momentum_term = Momentum_terms[num] / (1 - beta_Momentum ** t)
            RMSProp_term = RMSProp_terms[num] / (1 - beta_RMSProp ** t)
            param -= (lr/torch.sqrt(RMSProp_term + eps)) * Momentum_term
            param.grad.zero_()

#②设置参数
batch_size = 10
lr = 0.1
num_epochs = 20
beta_Momentum = 0.9
beta_RMSProp = 0.99
#end②

w = torch.normal(0, 1, size=(features.shape[1], 1), requires_grad=True, dtype=torch.float32)
b = torch.zeros(1, requires_grad=True, dtype=torch.float32)
losses = torch.zeros(num_epochs)

params = [w, b]
RMSProp_terms = [torch.zeros_like(param) for param in params]
Momentum_terms = [torch.zeros_like(param) for param in params]

net = linreg
loss = squared_loss
eps = 1e-8
t = 0

for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        l = loss(net(x, w, b), y)
        l.sum().backward()
        Adam(params, lr, Momentum_terms, RMSProp_terms, beta_Momentum, beta_RMSProp, t, batch_size, eps)

    with torch.no_grad():
        train_l = torch.sqrt(loss(net(features, w, b), labels)).mean()
        print(f'epoch {epoch + 1}, loss {float(train_l):f}')
        losses[epoch] = train_l

w = torch.div(w, features_stds.reshape(w.shape))
b = b - torch.sum(torch.mul(w, features_means.reshape(w.shape)))

print(f'w: {w.reshape(1, -1)}')
print(f'b: {b}')

epochs = torch.arange(1, num_epochs+1)
# 折线图
plt.figure(figsize=(6, 3))
plt.plot(epochs, losses, color='b', linestyle='-', marker='.', ms=10, mec='b', linewidth=2)
# 设置x、y轴
plt.grid()
plt.xlim(0, num_epochs+1)
plt.xticks(torch.arange(0, num_epochs+1, 5))
#plt.yscale('log')
plt.xlabel("epoch")
plt.ylabel("loss")
# 展示图片
plt.title('loss vs. epochs')
plt.show(block=False)

serials = torch.arange(outputs.shape[0])
results = net(inputs, w, b).detach().numpy()

# 创建图形
plt.figure(figsize=(8, 6))
# 绘制第一个数据集的散点图
plt.scatter(serials, outputs.reshape(serials.shape), label='Actual Value', color='red', s=5)
# 绘制第二个数据集的散点图
plt.scatter(serials, results.reshape(serials.shape), label='Predict Value', color='blue', s=5)
#填充误差区域
plt.fill_between(serials, outputs.reshape(serials.shape), results.reshape(serials.shape), facecolor='gray', alpha=0.5, label='loss')
# 添加图例
plt.legend()
# 设置标题和轴标签
plt.xlabel('serials')
plt.ylabel('prices')
# 显示图形
plt.title('Predict Value vs. Actual Value')
plt.show()
