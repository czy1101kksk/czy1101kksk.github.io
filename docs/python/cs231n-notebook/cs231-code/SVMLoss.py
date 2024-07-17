import numpy as np

# x:得分矩阵 [[3.2, 5.1, -1.7],
    #            [1.3, 4.9, 2,0],
    #            [2.2, 2.5, -3.1],
    #            ..]
    #y:真实标签 [0, 1, 2]
    
class MulticlassSVMLoss:
    def __init__(self, x, y, delta):
        self.x_train = x
        self.y_train = y
        self.delta = delta
    def train(self):
        x_true = np.array([self.x_train[k][self.y_train[k]] for k in range(len(self.y_train))])
        margins = np.maximum(self.x_train - x_true.reshape(-1,1) + self.delta, 0)
        loss = np.sum(margins,axis=1) - 1
        
        return np.sum(loss) / len(self.y_train)
    
x = [[3.2, 5.1, -1.7],
     [1.3, 4.9, 2.0],
     [2.2, 2.5, -3.1]]
y = [0, 1, 2]

print(MulticlassSVMLoss(x, y, 1).train())