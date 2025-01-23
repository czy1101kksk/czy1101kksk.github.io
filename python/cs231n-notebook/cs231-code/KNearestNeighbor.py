import numpy as np
import collections

# input x.shape [num_examples, y, x]

#  [ [[1, 2, 3],
#     [2, 3, 4],
#     [4, 5, 6]],
#
#    [[1, 2, 3],
#     [2, 3, 4],
#     [4, 5, 6]] ]

class NearestNeighbor:
    def __init__(self):
        pass
    
    def train(self, X, Y):
        self.X_train = X
        self.y_train = Y
    
    def distance(self, x_1, x_2):
        dis = np.sum(np.abs(x_1 - x_2), axis=1)
        return np.sum(dis, axis=1)

    def predict(self, X):
        num_examples = X.shape[0]
        y_predict = np.zeros(num_examples, dtype=self.y_train.dtype)
        for i in range(num_examples):
            Dis_p = self.distance(self.X_train, X[i,:])
            min_index = np.argmin(Dis_p)
            y_predict[i] = self.y_train[min_index]
        return y_predict

class KNearestNeighbor:
    def __init__(self, k):
        super().__init__()
        self.k = k
    
    def train(self, X, Y):
        self.X_train = X
        self.y_train = Y

    def distance(self, x_1, x_2):
        dis = np.sum(np.abs(x_1 - x_2), axis=1)
        return np.sum(dis, axis=1)
    
    def predict(self, X):
        num_examples = X.shape[0]
        y_predict = np.zeros(num_examples, dtype=self.y_train.dtype)
        for i in range(num_examples):
            Distance = self.distance(self.X_train, X[i,:])
            sorted_indices = np.argsort(Distance)
            k_nearest_labels = self.y_train[sorted_indices]
            y_predict[i] = collections.Counter(k_nearest_labels[0:self.k]).most_common(1)[0][0]

        return y_predict
        
            


if __name__ == '__main__':
    
    X_1 = np.random.random(size=(10, 3, 3))
    y_1 = np.array([1, 2, 3, 2, 1, 2, 3, 1, 2, 1])
    X_pre = np.random.random(size=(6, 3, 3))

    KNearN = KNearestNeighbor(k=3)
    KNearN.train(X_1, y_1)
    y_pre = KNearN.predict(X_pre)
    print(y_pre)