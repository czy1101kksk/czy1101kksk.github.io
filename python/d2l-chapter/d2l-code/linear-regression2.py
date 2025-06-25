import numpy as np
import torch
from torch.utils import data
from torch import nn

def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2.0, -3.4, 5.5, -6.9, 7])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
batch_size = 10

dataset = data.TensorDataset(features, labels)
dataLoader = data.DataLoader(dataset, batch_size, shuffle=True)

net = nn.Sequential(nn.Linear(5, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss(reduction='mean')
trainer = torch.optim.SGD(net.parameters(), lr=0.05)

num_epochs = 30
for epoch in range(num_epochs):
    for X, y in dataLoader:
        l = loss(net(X) ,y) 
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels) 
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)