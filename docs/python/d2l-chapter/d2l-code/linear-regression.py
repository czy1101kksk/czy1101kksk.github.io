import math
import torch
import random

def LinearRegression(X, w, b):
    y_hat = torch.matmul(w, X.T) + b 
    return y_hat.reshape(-1, 1) 

def MSELossfunction(y_hat, y):
    return (y - y_hat) ** 2

def Dataiter_RandomBatch(batch_size, features, labels):   # 迭代器
    num = len(features)
    numlist = [i for i in range(0, num)]
    random.shuffle(numlist)
    for k in range(0, num, batch_size):
        RandomBatch = numlist[k:min(k + batch_size, num)]
        yield features[RandomBatch], labels[RandomBatch]

def CreatData(features, num_examples, w, b):
    X = torch.normal(0, 1, size = (num_examples, features))
    y = torch.matmul(w, X.T) + b + torch.normal(0, 0.1, size=(1, num_examples))
    return X, y.reshape((-1,1))

def SGD(params, alpha, batch_size):
    with torch.no_grad():
        for param in params:
            param -= alpha * param.grad / batch_size
            param.grad.zero_()

w_0 = torch.tensor([1, 3, 2, 4, 5, 6], dtype=torch.float)
b_0 = 5.5

num_examples = 100 
features = len(w_0)

X,y = CreatData(features, num_examples, w_0, b_0)

w = torch.normal(mean = 0, std = 1, size = w_0.shape, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

num_epochs = 50
alpha = 0.03
batch_size = 10

for epoch in range(num_epochs):    
    for X_batch, y_batch in Dataiter_RandomBatch(batch_size, X, y):
        loss = MSELossfunction(y_batch, LinearRegression(X_batch, w, b))
        loss.backward()
        SGD([w, b], alpha, batch_size)  
    with torch.no_grad():
        train_loss = MSELossfunction(LinearRegression(X, w, b), y)
        print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')


print(f'w的估计误差: {w_0 - w}')
print(f'b的估计误差: {b_0 - b}')