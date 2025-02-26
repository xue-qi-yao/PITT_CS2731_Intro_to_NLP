import numpy as np

class logistic_regression():
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.w = np.zeros((self.y.shape[1], self.x.shape[1]))
        self.b = np.zeros((self.y.shape[1], 1))
    
    def cal_loss(self, pred, label):
        return -np.sum((label * np.log(pred) + (1-label) * np.log(1-pred))) / label.shape[0]
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, epoch_num=30, lr=0.2):
        self.lr = lr
        for epoch in range(epoch_num):
            self.logit = self.w @ self.x.T + self.b
            self.pred = self.sigmoid(self.logit)
            self.loss = self.cal_loss(self.pred, self.y.T)
            self.w -= self.lr / self.x.shape[0] * np.sum((self.pred - self.y.T) * self.x.T,  axis=-1)
            self.b -= self.lr / self.x.shape[0] * np.sum((self.pred - self.y.T), axis=-1)
            print(f"epoch {epoch} | loss {self.loss} | w {self.w} | b {self.b}")
    
    def test(self, x):
        return np.where(self.sigmoid(self.w @ np.array(x).T + self.b)>=0.5, 1, 0).T

if __name__ == "__main__":
    x = [[2, 1],
         [1, 3],
         [0, 4]]
    y = [[1], 
         [0],
         [0]]
    logit_regress = logistic_regression(x, y)
    logit_regress.train()
    test_result = logit_regress.test(x)
    print(test_result)