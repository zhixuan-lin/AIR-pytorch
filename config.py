import os
from attrdict import AttrDict
cfg = AttrDict({
    'exp_name': 'test',
    'resume': True,
    'device': 'cuda:5',
    'train': {
        'batch_size': 64,
        'model_lr': 1e-4,
        'baseline_lr': 1e-1,
        'max_epochs': 1000
    },
    'valid': {
        'batch_size': 64
    },
    'logdir': 'logs/',
    'checkpointdir': 'checkpoint/',
    'multi_mnist_path': os.path.join('data', 'multi_mnist')
})
