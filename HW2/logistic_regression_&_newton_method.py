import scipy.io as sio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_data(path):
    return sio.loadmat(path)

class LogisticRegression():

    def __init__(self, x, y, lr, num_iter, th, file_name):
        # x.shape: (m, n), y.shape:(1, n)
        self.x = x
        self.y = y
        self.lr = lr
        self.num_iter = num_iter
        self.num = y.shape[1]
        self.th = th
        self.file_name = file_name

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def train(self):
        m = self.x.shape[0]
        n = self.x.shape[1]

        X_tilde = np.vstack([self.x, np.ones((1, n))])

        theta = np.zeros((m+1, 1))

        for i in range(self.num_iter):
            z = theta.T @ X_tilde  
            p = self._sigmoid(z)
            
            loss = -np.sum(self.y * np.log(p) + (1 - self.y) * np.log(1 - p)) / n
            
            pred_result = np.where(p >= self.th, 1, 0)
            acc = np.sum(pred_result == self.y) / n
            print(f"Round {i} | Avg loss: {loss:.4f} | Accuracy: {acc}")
            
            g = (X_tilde @ (p - self.y).T) / n

            W = np.diag((p * (1 - p)).flatten())
            H = (X_tilde @ W @ X_tilde.T) / n

            delta = np.linalg.solve(H, g)
            theta = theta - delta

        self.a = theta[:-1].T
        self.b = theta[-1].reshape(1, 1)

        self._plot_decision_boundary()

    def _plot_decision_boundary(self):
        plt.figure(figsize=(8, 6))
        # Scatter the two classes.
        plt.scatter(self.x[0, self.y[0] == 0], self.x[1, self.y[0] == 0], 
                    color='blue', label='Class 0', edgecolors='k')
        plt.scatter(self.x[0, self.y[0] == 1], self.x[1, self.y[0] == 1], 
                    color='red', label='Class 1', edgecolors='k')
        start = np.min(self.x[0])
        end  = np.max(self.x[0])
        x_vals = np.linspace(start, end, 100)
        if self.a.shape[1] >= 2 and self.a[0, 1] != 0: 
            y_vals = (self.th - (self.a[0, 0] * x_vals + self.b) / self.a[0, 1]).squeeze()
            plt.plot(x_vals, y_vals, color='green', linestyle='--', label='Decision Boundary')

        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title(f"Decision Boundary on {self.file_name}")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def test(self):
        self._pred()
        pred_result = np.where(self.pred >= self.th, 1, 0)
        self.acc = np.sum(pred_result == self.y) / self.num

    def _pred(self):
        # return shape (1, n)
        self.pred = self._sigmoid(self.a @ self.x + self.b)

    def _loss(self):
        self.loss = -np.sum(self.y * np.log(self.pred) + (1-self.y) * np.log(1-self.pred)) / self.num
    

if __name__ == "__main__":
    file_dicts = {}
    path = Path("dataLDA")
    # set learning rate at 0.001, epoch at 30
    learning_rate = 0.001
    threshold = 0.5
    epoch = 30
    for file_path in path.rglob("synthetic*.mat"):
        file_dict = load_data(file_path)
        file_dicts[file_path.stem] = file_dict
    
    for file_name, file_dict in file_dicts.items():
        x = np.array(file_dict["X"])
        y = np.array(file_dict["Y"])
        lr = LogisticRegression(x, y, learning_rate, epoch, threshold, file_name)
        print(f"\n*** Start training on dataset {file_name}\n")
        lr.train()

# The synthetic data 4 accuracy converge in 2 rounds
# The synthetic data 1 accuracy converge in 2 rounds
# The synthetic data 3 accuracy converge in 3 rounds
# The synthetic data 2 accuracy converge around 4 rounds