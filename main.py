import torch
from torch.utils.data import DataLoader
from multi_mnist import MultiMNIST
import os
from attrdict import AttrDict

cfg = AttrDict({
    'exp_name': 'test',
    'train': {
        'batch_size': 32
    },
    'valid': {
        'batch_size': 32
    },
    'multi_mnist_path': os.path.join('data', 'multi_mnist')
})

if __name__ == '__main__':
    trainset = MultiMNIST(path=cfg.multi_mnist_path, mode='train')
    validset = MultiMNIST(path=cfg.multi_mnist_path, mode='test')
    trainloader = DataLoader(trainset, batch_size=cfg.train.batch_size,
                             shuffle=True, num_workers=4)
    validloader = DataLoader(validset, batch_size=cfg.test.batch_size,
                             shuffle=False, num_workers=4)
