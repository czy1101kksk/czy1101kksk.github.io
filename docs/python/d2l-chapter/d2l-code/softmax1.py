import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms

def evaluate_accuracy(net, iter):
    cmp = 0
    tot = 0
    net.eval() # 将模型设置为评估模式
    with torch.no_grad():
        for X, y in iter:
            y_hat = net(X)
            y_hat = y_hat.argmax(axis=1)  # 在行中比较，选出最大的列索引
            CountMatrix = y_hat == y
            cmp += sum(CountMatrix)
            tot += len(CountMatrix)
    return cmp / tot
            
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=transforms.ToTensor(), download=True)

# Fashion-MNIST由10个类别的图像组成,每个类别由训练数据集中的6000张图像和测试数据集中的1000张图像组成.

batch_size = 256

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=0)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=0)

net = nn.Sequential(nn.Flatten(), 
                    nn.Linear(784, 10))

def init_weights(module):   #权重初始化。apply()
    if isinstance(module,nn.Linear):
        nn.init.normal_(module.weight, std=0.01, mean=0)
        nn.init.constant_(module.bias, 0)

net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='mean')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10

for epoch in range(num_epochs):
    for X_train , y_train in train_iter:
        l =loss(net(X_train), y_train)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    print(f'{evaluate_accuracy(net, train_iter) * 100:.3f}%')
