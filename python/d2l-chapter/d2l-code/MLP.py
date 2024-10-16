import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256, 5)
    def forward(self, X_input):
        return self.out(F.relu(self.hidden(X_input)))

class Net(nn.Module):
        def __init__(self, net1, net2):
            super().__init__()
            self.Net_1 = net1
            self.Net_2 = net2
        def forward(self, X):
            X = torch.cat((self.Net_1(X), self.Net_2(X)), 1)
            return X

if __name__ == '__main__':
    X = torch.randn(5, 20, requires_grad=True)
    
    net = MLP()
    net0 = Net(nn.Linear(20,3), nn.Linear(20,6))
    
    y_1 = net(X)
    y_2 = net0(X)
    print(y_1)
    print(y_2)

