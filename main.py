import torch
torch.manual_seed(233)
import numpy as np
np.random.seed(233)
from torch.utils.data import DataLoader
from multi_mnist import MultiMNIST
import os
from attrdict import AttrDict
from air import AIR
from torch import optim
from utils import vis_logger, metric_logger, Checkpointer
from tensorboardX import SummaryWriter
from config import cfg
from torch import autograd


if __name__ == '__main__':
    trainset = MultiMNIST(path=cfg.multi_mnist_path, mode='train')
    # validset = MultiMNIST(path=cfg.multi_mnist_path, mode='test')
    trainloader = DataLoader(trainset, batch_size=cfg.train.batch_size,
                             shuffle=True, num_workers=4)
    # validloader = DataLoader(validset, batch_size=cfg.test.batch_size,
    #                          shuffle=False, num_workers=4)
    
    device = torch.device(cfg.device)
    model = AIR().to(device)
    optimizer = optim.Adam([
        {'params': model.air_modules.parameters(), 'lr': cfg.train.model_lr},
        {'params': model.baseline_modules.parameters(), 'lr': cfg.train.baseline_lr}
    ])
    
    
    # checkpoint
    start_epoch = 0
    checkpoint_path = os.path.join(cfg.checkpointdir, cfg.exp_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpointer = Checkpointer(path=checkpoint_path)
    if cfg.resume:
        start_epoch = checkpointer.load(model, optimizer)

    # tensorboard
    writer = SummaryWriter(logdir=os.path.join(cfg.logdir, cfg.exp_name))
    
    print('Start training')
    with autograd.detect_anomaly():
        for epoch in range(start_epoch, cfg.train.max_epochs):
            for i, (x, num) in enumerate(trainloader):
                global_step = epoch * len(trainloader) + i + 1
                # pred: (B,)
                x = x.to(device)
                num = num.to(device)
                loss, pred = model(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Evaluate accuracy
                zero_indices = num == 0
                one_indices = num == 1
                two_indices = num == 2
                correct = (pred == num)
                correct_zero = (correct & zero_indices)
                correct_one = (correct & one_indices)
                correct_two = (correct & two_indices)
                acc_total = correct.float().mean()
                acc_zero = correct_zero.float().sum() / (zero_indices.sum() + 1e-5)
                acc_one = correct_one.float().sum() / (one_indices.sum() + 1e-5)
                acc_two = correct_two.float().sum() / (two_indices.sum() + 1e-5)
                
                metric_logger.update(loss=loss.item())
                metric_logger.update(acc_total=acc_total.item())
                metric_logger.update(acc_zero=acc_zero.item())
                metric_logger.update(acc_one=acc_one.item())
                metric_logger.update(acc_two=acc_two.item())
                
                
                if (i + 1) % 50 == 0:
                    print('Epoch: {}/{}, Iter: {}/{}, Loss: {:.2f}, Acc: {:.2f}'.format(
                        epoch + 1, cfg.train.max_epochs, i + 1, len(trainloader), metric_logger['loss'].median,
                        metric_logger['acc_total'].median))
                    vis_logger.add_to_tensorboard(writer, global_step)
                    writer.add_scalar('accuracy/total', acc_total, global_step)
                    writer.add_scalar('accuracy/zero', acc_zero, global_step)
                    writer.add_scalar('accuracy/one', acc_one, global_step)
                    writer.add_scalar('accuracy/two', acc_two, global_step)
                    
            checkpointer.save(model, optimizer, epoch+1)
            
            
            
            
