import torch
import torchvision
from torch.utils import data
from torchvision import transforms

mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=transforms.ToTensor(), download=True)

# Fashion-MNIST由10个类别的图像组成,每个类别由训练数据集中的6000张图像和测试数据集中的1000张图像组成.

batch_size = 256

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=1)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=1)



net = nn.Sequential(nn.Flatten(), 
                    nn.Linear(784, 10))