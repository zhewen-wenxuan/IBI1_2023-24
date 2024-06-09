import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# 数据预处理
'''
定义了一个图像数据的预处理管道，使用的是PyTorch中的transforms模块。具体来说，它使用了两个转换操作：将图像转换为张量（ToTensor()）和对图像进行归一化（Normalize()）
Normalize是一个归一化操作，用于对图像的像素值进行标准化处理。它接收两个参数：mean和std，分别表示均值和标准差。这样处理后，图像的像素值将会在[-1, 1]范围内。
'''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载并加载训练集和测试集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  # 输入通道为3，输出通道为32，卷积核大小为3x3，步幅为1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 输入通道为32，输出通道为64，卷积核大小为3x3，步幅为1
        self.fc1 = nn.Linear(14 * 14 * 64, 128)  # 输入特征维度为14*14*64，输出特征维度为128
        self.fc2 = nn.Linear(128, 10)  # 输入特征维度为128，输出特征维度为10（分类数量）

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2) #池化层2*2进行降维，维度变为原来的1/2
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型，定义损失函数和优化器
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {loss.item():.4f}')

# 测试模型并可视化预测结果
def visualize_predictions():
    model.eval()
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    images, labels = images.to(device), labels.to(device)

    # 预测
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # 显示图片和预测结果
    fig = plt.figure(figsize=(12, 8))
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()
    for idx in range(8):
        ax = fig.add_subplot(2, 4, idx + 1, xticks=[], yticks=[])
        img = images[idx] / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(f'Label: {labels[idx].item()}\nPrediction: {predicted[idx].item()}')
    plt.savefig('/mnt/python_learn/6_1/8_CNN_predictions.png')  # 保存图像

# 训练完成后调用可视化函数
visualize_predictions()
plt.show()
