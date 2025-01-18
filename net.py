import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# 设置图像的预处理转换
transform = transforms.Compose(
    [
        # 将图像尺寸调整为 224x224，以符合 AlexNet 的输入要求
        transforms.Resize(224),

        # 将图像转换为 Tensor 格式，范围在 [0, 1] 之间
        transforms.ToTensor(),

        # 对图像进行标准化，使用均值和标准差来标准化每个通道的值（RGB）
        # 这里均值和标准差设置为 [0.5, 0.5, 0.5]，以将值映射到 [-1, 1] 之间
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)

# 加载训练集（CIFAR-10）
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
# 使用 DataLoader 来加载数据，batch_size=4 表示每批次加载 4 张图片，shuffle=True 表示打乱数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# 加载测试集（CIFAR-10）
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
# 使用 DataLoader 加载测试数据
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# 定义 CIFAR-10 数据集的类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 创建 AlexNet 模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        # 特征提取部分（卷积层 + 激活层 + 池化层）
        self.features = nn.Sequential(
            # 第一层卷积层，输入通道为 3（RGB 图像），输出通道为 64，卷积核大小为 11x11，步幅为 4，填充为 2
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化，池化核大小为 3x3，步幅为 2

            # 第二层卷积层，输入通道为 64，输出通道为 192，卷积核大小为 5x5，填充为 2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第三层卷积层，输入通道为 192，输出通道为 384，卷积核大小为 3x3，填充为 1
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第四层卷积层，输入通道为 384，输出通道为 256，卷积核大小为 3x3，填充为 1
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第五层卷积层，输入通道为 256，输出通道为 256，卷积核大小为 3x3，填充为 1
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 最大池化层，用于减少图像的空间维度，池化核大小为 3x3，步幅为 2
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 自适应平均池化，将输出大小固定为 6x6（用于全连接层）
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # 分类器部分（全连接层 + Dropout）
        self.classifier = nn.Sequential(
            nn.Dropout(),  # Dropout，防止过拟合
            nn.Linear(256 * 6 * 6, 4096),  # 全连接层，将输入 256*6*6 展平后送入，输出 4096
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # 第二个全连接层，输出 4096
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # 最后一层全连接层，输出类别数（CIFAR-10 为 10）
        )

    def forward(self, x):
        # 前向传播：数据通过卷积层、池化层提取特征，再通过全连接层进行分类
        x = self.features(x)  # 特征提取
        x = self.avgpool(x)  # 自适应平均池化
        x = torch.flatten(x, 1)  # 将多维的张量展平为一维，传递给全连接层
        x = self.classifier(x)  # 分类
        return x


# 实例化模型
net = AlexNet()
print(net)  # 打印模型结构


def train_and_test():
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，用于多分类任务
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 随机梯度下降优化器，学习率为 0.001，动量为 0.9

    # 训练模型
    for epoch in range(10):  # 训练 10 个周期
        running_loss = 0.0  # 初始化累计损失
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data  # 获取输入数据和标签
            optimizer.zero_grad()  # 清空梯度
            outputs = net(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
            running_loss += loss.item()  # 累加损失值
            if i % 2000 == 1999:  # 每 2000 个小批量打印一次损失
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0  # 重置累计损失

    # 在测试集上进行测试
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for data in testloader:
            images, labels = data  # 获取测试集图像和标签
            outputs = net(images)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # 取最大值的索引作为预测类别
            total += labels.size(0)  # 总的样本数
            correct += (predicted == labels).sum().item()  # 统计正确预测的样本数
    print('在10000张测试图像上的准确率: %d %%' % (100 * correct / total))  # 输出准确率


# 入口函数，执行训练和测试
if __name__ == '__main__':
    train_and_test()
