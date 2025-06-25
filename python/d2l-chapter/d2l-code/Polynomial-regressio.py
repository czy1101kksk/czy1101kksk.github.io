import math
import numpy as np
import torch
from torch import nn
from torch.utils import data 


features = np.random.normal(size=(2000, 1))
poly_features = np.power(features, np.arange(4).reshape(1, -1))

for i in range(4):
    poly_features[:, i] /= math.gamma(i + 1)

true_w = np.array([5, 1.2, -3.4, 5.6])

labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.01, size=labels.shape)

true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]

net = nn.Sequential(nn.Linear(4, 1, bias=False))
net[0].weight.data.normal_(0, 0.01)

loss = nn.MSELoss(reduction='mean')
trainer = torch.optim.SGD(net.parameters(), lr=0.01)

batch_size = 10
polynomial_dataset = data.TensorDataset(poly_features, labels)
polynomial_dataiter = data.DataLoader(polynomial_dataset, batch_size, shuffle=True)

for epoch in range(100):
    for X_train, y_train in polynomial_dataiter:
        l = loss(net(X_train), y_train)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(poly_features), labels) 
    print(f'epoch {epoch + 1}, loss {l:.3f}')

print(net[0].weight.data)

