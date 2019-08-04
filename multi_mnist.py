from observations import multi_mnist
import torch
import numpy as np
from torch.utils.data import Dataset


class MultiMNIST(Dataset):
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        (X_train, y_train), (X_test, y_test) = multi_mnist(path)
        if self.mode == 'train':
            self.X, self.y = X_train, y_train
        else:
            self.X, self.y = (X_test, y_test)
        
    def __getitem__(self, index):
        """
        Returns (X, y), where X is (H, W) in range (0, 1), y is the number of
        digits.
        """
        # x: uint8, (H, W)
        # y: ndarray: label
        x, y = self.X[index], self.y[index]
        y = np.array(len(y))
        x = x / 255.0
        
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        
        return x, y
        
        
    def __len__(self):
        return len(self.X)
