import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
'''
torch：PyTorch的核心库。
torch.nn：包含神经网络相关的模块。
torch.optim：包含各种优化算法。
torch.utils.data：数据加载和处理相关的模块。
torchvision：提供了常用的数据集和图像变换。
'''
# 定义超参数
batch_size = 32
learning_rate = 0.001
epochs = 5

'''
数据预处理非常复杂，根据实际需求进行设置，并且可能需要我们单独用算法进行实现。
将图像转换为张量并归一化，使其均值为0.1307，标准差为0.3081。
'''
# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


'''
class SimpleNN(nn.Module)：定义一个名为SimpleNN的类，继承自torch.nn.Module。nn.Module是所有神经网络模块的基类。

def __init__(self):：定义类的构造函数。这个函数在创建类的实例时自动调用。

super(SimpleNN, self).__init__()：调用父类nn.Module的构造函数，初始化模块的基本结构。这是必需的步骤，确保我们的自定义模块可以正确地集成到PyTorch的框架中。

self.flatten = nn.Flatten()：定义一个展平层。nn.Flatten()会将多维输入张量展平成一维向量。
例如，一个形状为[batch_size, 28, 28]的输入张量会被展平成[batch_size, 28*28]的形状。

self.fc1 = nn.Linear(28 * 28, 128)：定义一个全连接层fc1。这个层将输入的展平向量（长度为28*28=784）映射到一个长度为128的向量。
nn.Linear表示一个全连接层，参数28 * 28是输入维度，128是输出维度。

self.relu = nn.ReLU()：定义一个ReLU激活函数。ReLU（Rectified Linear Unit）是一个常用的激活函数，将输入中的负值置为零，正值保持不变。
它引入了非线性，使得神经网络能够学习复杂的模式。

self.fc2 = nn.Linear(128, 10)：定义第二个全连接层fc2。这个层将长度为128的向量映射到长度为10的向量。
10是输出的类别数量，对应MNIST数据集的10个数字（0到9）。

self.softmax = nn.Softmax(dim=1)：定义一个Softmax激活函数。Softmax函数将输出向量中的每个元素变换为0到1之间的概率值，并且这些概率值的和为1。
参数dim=1表示对每个样本的输出向量进行Softmax操作。最后转变为0-1的概率模式输出。

def forward(self, x):：定义前向传播方法forward。在训练和推理过程中，数据通过网络时会自动调用这个方法。
x = self.flatten(x)：将输入展平。
x = self.fc1(x)：将展平的输入通过第一个全连接层。
x = self.relu(x)：应用ReLU激活函数。
x = self.fc2(x)：将通过ReLU后的向量通过第二个全连接层。
x = self.softmax(x)：应用Softmax激活函数，输出每个类别的概率。
return x：返回输出结果。

这段代码定义了一个简单的前馈神经网络，它包含以下层次：

展平层：将输入的28x28像素的图像展平成一维向量。
第一个全连接层：将展平的向量映射到长度为128的向量。
ReLU激活函数：引入非线性。
第二个全连接层：将长度为128的向量映射到长度为10的向量。
Softmax激活函数：将输出转换为概率分布。
'''


# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


'''
model = SimpleNN()：实例化我们定义的神经网络模型SimpleNN。
这一步将创建一个SimpleNN类的对象，并初始化其所有定义的层和参数。
SimpleNN继承自nn.Module，所以它是一个PyTorch模型。
在实例化时，调用了SimpleNN类的构造函数__init__()，这会初始化展平层、全连接层、ReLU激活函数和Softmax激活函数。

criterion = nn.CrossEntropyLoss()：定义损失函数，使用交叉熵损失（Cross-Entropy Loss）。
交叉熵损失函数通常用于分类问题，特别是多分类任务。
交叉熵损失函数计算输出与目标之间的差异，衡量预测的类别概率分布与真实类别分布之间的距离。目标是最小化这个损失，从而提高模型的分类准确性。

'''

# 实例化模型、损失函数和优化器
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

'''
model.train()：将模型设置为训练模式。这会启用训练模式中的特定功能，比如激活函数的训练和Dropout层的启用。

for batch_idx, (data, target) in enumerate(train_loader):：遍历训练数据加载器中的每个批次。
enumerate(train_loader)：将train_loader中的数据和对应的索引进行枚举，其中data是输入数据，target是对应的标签。
batch_idx：批次的索引，从0开始计数。

optimizer.zero_grad()：清空之前的梯度，防止梯度累积。
loss.backward()：反向传播，计算损失函数关于模型参数的梯度。
optimizer.step()：根据梯度更新模型参数，执行优化步骤。
'''
# 训练模型
for epoch in range(epochs):
    model.train()  #将模型设置为训练模式。
    for batch_idx, (data, target) in enumerate(train_loader):
        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')


'''
model.eval()：将模型设置为评估模式。这会禁用特定于训练的功能，例如Dropout层和批量归一化的更新，并启用推断时需要的特定功能。

with torch.no_grad():：使用torch.no_grad()上下文管理器，这样在下面的代码块中计算的梯度不会被保存。因为我们在评估阶段不需要计算梯度，只需要进行前向传播和计算准确率。

torch.max(output.data, 1) 是一个PyTorch函数，用于在张量中沿着指定维度（这里是维度1，即每行）找到最大值及其对应的索引。 
返回的第一个张量将是每个样本的预测概率中的最大值，而第二个张量将是每个样本预测的类别索引。我们仅仅需要对比类别就可以，不需要考虑值是什么情况。

total += target.size(0)：将当前批次的样本数累加到总样本数中。每个target.size(0)都是当前批次中的样本数。

correct += (predicted == target).sum().item()：统计模型在当前批次中预测正确的样本数。
'''
# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1) # _是最大的值，不重要，我们仅仅需要predict预测的类别输出即可，看看是什么类型与target对比看看是否一致。
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')
