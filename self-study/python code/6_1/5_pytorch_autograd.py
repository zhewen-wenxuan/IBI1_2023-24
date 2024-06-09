import torch
import torch.nn as nn
import torch.optim as optim

'''
loss.backward() 是利用 autograd 自动计算损失函数对模型参数的导数的关键步骤。
Autograd 会根据计算图自动追踪每个操作的梯度流，并根据链式法则计算出损失函数对模型参数的导数。
然后，优化器使用这些导数信息来更新模型参数，从而使得损失函数达到最小值。

'''
# 定义一个简单的神经网络类
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 5)  # 输入层到隐藏层的全连接层
        self.fc2 = nn.Linear(5, 1)  # 隐藏层到输出层的全连接层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU作为激活函数
        x = self.fc2(x)
        return x


# 创建神经网络实例
model = SimpleNet()

# 创建一些示例数据和标签
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
labels = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 使用随机梯度下降优化器

# 迭代训练
for epoch in range(1000):
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每100次迭代打印一次损失
    if epoch % 100 == 0:
        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')

# 使用训练好的模型进行预测
with torch.no_grad():
    test_input = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    test_labels = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    # 前向传播得到预测值
    predictions = model(test_input)

    # 计算损失
    loss = criterion(predictions, test_labels)

    # 计算准确率
    correct = (torch.round(predictions) == test_labels).sum().item()
    total = test_labels.size(0)
    accuracy = correct / total

    print(f"Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%")

