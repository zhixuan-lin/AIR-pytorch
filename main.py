import torch
from torch.utils.data import DataLoader
from multi_mnist import MultiMNIST
import os
from attrdict import AttrDict
from air import AIR
from torch import optim

cfg = AttrDict({
    'exp_name': 'test',
    'train': {
        'batch_size': 64,
        'model_lr': 1e-4,
        'baseline_lr': 1e-3,
    },
    'valid': {
        'batch_size': 64
    },
    'multi_mnist_path': os.path.join('data', 'multi_mnist')
})

if __name__ == '__main__':
    trainset = MultiMNIST(path=cfg.multi_mnist_path, mode='train')
    # validset = MultiMNIST(path=cfg.multi_mnist_path, mode='test')
    trainloader = DataLoader(trainset, batch_size=cfg.train.batch_size,
                             shuffle=True, num_workers=4)
    # validloader = DataLoader(validset, batch_size=cfg.test.batch_size,
    #                          shuffle=False, num_workers=4)
    
    model = AIR()
    optimizer = optim.Adam([
        {'params': model.air_modules.parameters(), 'lr': cfg.train.model_lr},
        {'params': model.baseline_modules.parameters(), 'lr': cfg.train.baseline_lr}
    ])
    
    for epoch in range(100):
        for i, (x, num) in enumerate(trainloader):
            
            loss = model(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
