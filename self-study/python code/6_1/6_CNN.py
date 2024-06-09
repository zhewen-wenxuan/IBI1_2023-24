import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np  # 导入 numpy

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 下载并加载训练集和测试集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

'''

self.conv1 = nn.Conv2d(1, 32, 3, 1) 是在 PyTorch 中定义卷积层的语句。下面是对该卷积层参数的解释：

1：输入通道数（in_channels）。对于 MNIST 数据集，输入图像是灰度图像，只有一个通道（黑白），所以这里是 1。
32：输出通道数（out_channels）。这是卷积层应用于输入图像时生成的特征图（feature map）的数量，也可以理解为滤波器的数量。每个滤波器会产生一个特征图，所以这里有 32 个特征图。
3：卷积核的尺寸（kernel_size）。这是一个 3x3 的卷积核。
1：步幅（stride）。步幅决定了卷积核在图像上滑动的步长。步幅为 1 表示卷积核每次移动一个像素。

更详细的解释
输入通道数（in_channels）
表示输入图像的通道数。例如，对于彩色图像，这个值通常是 3（对应 RGB 三个通道）；对于灰度图像，这个值是 1。

输出通道数（out_channels）
表示卷积层的输出通道数，即卷积层应用到输入图像上后生成的特征图的数量。更多的输出通道数通常表示更高的特征提取能力。

卷积核尺寸（kernel_size）
表示卷积核的宽度和高度。卷积核是一个小的矩阵，在输入图像上滑动并执行卷积操作。卷积核的尺寸通常是奇数（如 3x3、5x5 等）。

步幅（stride）
表示卷积核在输入图像上每次滑动的步长。步幅为 1 表示卷积核每次移动一个像素；步幅为 2 表示卷积核每次移动两个像素，依此类推。较大的步幅会减少输出特征图的尺寸。
'''
# 定义简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 实例化模型，定义损失函数和优化器
model = SimpleCNN().to(device)  # 将模型移动到 GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到 GPU
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {loss.item():.4f}')


'''
model.eval():
将模型设置为评估模式。这会禁用训练中使用的某些特性（例如 dropout 和 batch normalization,并且grad和autograd都禁用，没必要进行backward），以确保评估时的结果是确定的。

dataiter = iter(testloader):
创建一个迭代器，用于从测试数据加载器 (testloader) 中按批次获取数据。

images, labels = dataiter.next():
从迭代器中获取下一批数据，包括图像 (images) 和对应的标签 (labels)。

outputs = model(images):
将获取的图像输入到模型中，得到模型的输出（通常是每个类别的得分）。

_, predicted = torch.max(outputs, 1):
对模型的输出进行处理，得到每个样本的预测标签。torch.max(outputs, 1) 返回每行的最大值及其索引，这里我们只需要索引（即预测标签）。

fig = plt.figure(figsize=(12, 8)):
创建一个新的图像（figure），并设置其大小（宽度 12 英寸，高度 8 英寸）。

for idx in range(8):
遍历前 8 个图像。我们只显示前 8 个图像的预测结果。

ax = fig.add_subplot(2, 4, idx + 1, xticks=[], yticks=[]):
在图像中添加子图。这里设置了一个 2 行 4 列的网格，第 idx + 1 个子图。xticks=[] 和 yticks=[] 表示不显示刻度。

img = images[idx] / 2 + 0.5:
将图像数据反归一化。训练过程中图像数据通常会被归一化到 [-1, 1]，这里将其转回到 [0, 1]

npimg = img.numpy():
将图像张量转换为 numpy 数组，以便使用 matplotlib 进行显示。

plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray'):
使用 matplotlib 显示图像。np.transpose(npimg, (1, 2, 0)) 将图像的维度从 (C, H, W) 转换为 (H, W, C)，这是 matplotlib 显示图像所需的格式。
为什么转换？
transforms.ToTensor():
ToTensor是一个转换操作，将PIL图像或numpy数组转换为PyTorch的张量（tensor）。具体来说，它将图像的像素值从0-255范围内的整数转换为0-1范围内的浮点数。
此外，它还会将图像的维度从(H, W, C)（高度，高度，通道）转换为(C, H, W)，这是PyTorch张量的默认格式。
cmap='gray' 指定使用灰度颜色图。

意思是训练模型，我需要将numpy变为tensor，同时图像的维度从(H, W, C)（高度，高度，通道）转换为(C, H, W)，之后训练结束，需要变为numpy进行可视化操作，反过来即可。


ax.set_title(f'Label: {labels[idx].item()}\nPrediction: {predicted[idx].item()}'):
设置子图的标题，显示该图像的真实标签和预测标签。
'''
# 预测和可视化结果
def visualize_predictions():
    model.eval()  # 设置模型为评估模式。这会禁用 dropout 和 batch normalization 等特性。

    dataiter = iter(testloader)  # 创建一个测试数据加载器的迭代器。
    images, labels = next(dataiter) # 从迭代器中获取下一个批次的图像和标签。
    images, labels = images.to(device), labels.to(device)  # 将数据移动到 GPU

    # 预测
    outputs = model(images)  # 将图像传入模型，得到输出（未归一化的得分）。
    _, predicted = torch.max(outputs, 1)  # 对输出进行处理，得到每个样本的预测标签。torch.max(outputs, 1) 返回每行的最大值及其索引，这里我们只需要索引，即预测标签。

    # 显示图片和预测结果
    fig = plt.figure(figsize=(12, 8))  # 创建一个图像，设置图像的尺寸（宽12英寸，高8英寸）。
    images = images.cpu()  # 将数据移动回 CPU，以便进行可视化
    labels = labels.cpu()
    predicted = predicted.cpu()

    for idx in range(8):  # 遍历前8个图像进行显示和预测结果的可视化。
        ax = fig.add_subplot(2, 4, idx + 1, xticks=[], yticks=[])  # 在图像中添加子图。这里是2行4列的网格，第idx+1个子图。
        img = images[idx] / 2 + 0.5  # 反归一化图像数据。原图像在预处理时归一化到 [-1, 1]，这里将其转回 [0, 1]。
        npimg = img.numpy()  # 将图像张量转换为 numpy 数组。
        plt.imshow(np.transpose(npimg, (1, 2, 0)),
                   cmap='gray')  # 使用 matplotlib 显示图像。np.transpose(npimg, (1, 2, 0)) 将图像的维度从 (C, H, W) 转换为 (H, W, C)，这是 matplotlib 显示图像所需的格式。
        ax.set_title(f'Label: {labels[idx].item()}\nPrediction: {predicted[idx].item()}')  # 设置子图的标题，显示真实标签和预测标签。
    plt.savefig('/mnt/python_learn/6_1/6_CNN_predictions.png')  # 保存图像

visualize_predictions()  # 调用函数进行可视化。
plt.show()  # 显示绘制的图像。
