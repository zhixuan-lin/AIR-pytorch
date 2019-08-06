from collections import defaultdict, deque
import pickle
from attrdict import AttrDict
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('agg')
from matplotlib import patches
from matplotlib import pyplot as plt


class VisLogger:
    """
    Global visualization logger
    """
    def __init__(self):
        self.things = {}
        
    def update(self, **kargs):
        for key, value in kargs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            self.things.update(value)
            
    def __getitem__(self, key):
        return self.things[key]
    
    def __setitem__(self, key, item):
        self.things[key] = item
        
    def add_to_tensorboard(self, writer: SummaryWriter, global_step):
        """
        Process data and return a dictionary
            - x
            vis_logger['z_pres_p_list'] = []
            vis_logger['z_pres_list'] = []
            vis_logger['canvas_list'] = []
            vis_logger['z_where_list'] = []
            vis_logger['object_enc_list'] = []
            vis_logger['object_dec_list'] = []
        """
        
        # losses
        kl_pres = torch.sum(torch.tensor(self.things['kl_pres_list'])).item()
        kl_where = torch.sum(torch.tensor(self.things['kl_where_list'])).item()
        kl_what = torch.sum(torch.tensor(self.things['kl_what_list'])).item()
        
        kl_total = self.things['kl_loss']
        baseline_loss = self.things['baseline_loss']
        reinforce_loss = self.things['reinforce_loss']
        neg_likelihood = self.things['neg_likelihood']
        
        writer.add_scalar('kl/kl_pres', kl_pres, global_step)
        writer.add_scalar('kl/kl_where', kl_where, global_step)
        writer.add_scalar('kl/kl_what', kl_what, global_step)
        writer.add_scalar('loss/kl_total', kl_total, global_step)
        writer.add_scalar('loss/baseline_loss', baseline_loss, global_step)
        writer.add_scalar('loss/reinforce_loss', reinforce_loss, global_step)
        writer.add_scalar('loss/neg_likelihood', neg_likelihood, global_step)
        
        canvas_list = [x.detach().cpu().numpy()[0] for x in self.things['canvas_list']]
        obj_enc_list = [x.detach().cpu().numpy()[0] for x in self.things['object_enc_list']]
        obj_dec_list = [x.detach().cpu().numpy()[0] for x in self.things['object_dec_list']]
        z_pres_list = [x.detach().cpu().item() for x in self.things['z_pres_list']]
        z_prob_list = [x.detach().cpu().item() for x in self.things['z_pres_p_list']]
        z_where_list = [x.detach().cpu().numpy() for x in self.things['z_where_list']]
        
        image = self.things['image']
        writer.add_image('vis/original', image.detach(), global_step)
        fig = create_fig(image[0].detach().cpu().numpy(), canvas_list, obj_enc_list, obj_dec_list, z_pres_list, z_prob_list, z_where_list)
        writer.add_figure('vis/reconstruct', fig, global_step)
    
def create_fig(image, canvas_list, obj_enc_list, obj_dec_list, z_pres_list, z_prob_list, z_where_list):
    """
    All types should be numpy or python numbers.
    """
    fig = plt.figure()
    for i in range(3):
        ax = fig.add_subplot(4, 3, i + 1)
        ax.set_title('pres: {:.0f}, p: {:.2f}'.format(z_pres_list[i], z_prob_list[i] if i == 0 or z_pres_list[i-1] == 1 else 0))
        ax.set_axis_off()
        ax.imshow(image, cmap='gray')
        if z_pres_list[i] == 1:
            draw_bounding_box(ax, z_where_list[i], (50, 50), 'r')
        
    # canvas
    for i in range(3):
        ax = fig.add_subplot(4, 3, i + 4)
        ax.set_axis_off()
        ax.imshow(canvas_list[i], cmap='gray')
        if z_pres_list[i] == 1:
            draw_bounding_box(ax, z_where_list[i], (50, 50), 'r')
            
    for i in range(3):
        ax = fig.add_subplot(4, 3, i + 7)
        ax.set_axis_off()
        if z_pres_list[i] == 1:
            ax.imshow(obj_enc_list[i], cmap='gray')
        else:
            ax.imshow(np.ones_like(obj_enc_list[i]).astype(np.uint8) * 255, cmap='gray')

    for i in range(3):
        ax = fig.add_subplot(4, 3, i + 10)
        ax.set_axis_off()
        if z_pres_list[i] == 1:
            ax.imshow(obj_dec_list[i], cmap='gray')
        else:
            ax.imshow(np.ones_like(obj_dec_list[i]).astype(np.uint8) * 255, cmap='gray')
            
    return fig


def draw_bounding_box(ax: plt.Axes, z_where, size, color):
    """
    :param ax: matplotlib ax
    :param z_where: [s, x, y] s < 1
    :param size: output size, (h, w)
    """
    h, w = size
    s, x, y = z_where
    
    min, max = -1, 1
    h_box, w_box = h / s, w / s
    x_box = (-x / s - min) / (max - min) * w
    y_box = (-y / s - min) / (max - min) * h
    x_box -= w_box / 2
    y_box -= h_box / 2
    
    rect = patches.Rectangle((x_box, y_box), w_box, h_box, edgecolor=color, linewidth=3.0, fill=False)
    ax.add_patch(rect)

class SmoothedValue:
    """
    Record the last several values, and return summaries
    """
    def __init__(self, maxsize=20):
        self.values = deque(maxlen=maxsize)
        self.count = 0
        self.sum = 0.0
    
    def update(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.values.append(value)
        self.count += 1
        self.sum += value
        
    @property
    def median(self):
        return np.median(np.array(self.values))
    
    @property
    def global_avg(self):
        return self.sum / self.count
    
class MetricLogger:
    def __init__(self):
        self.values = defaultdict(SmoothedValue)
        
    def update(self, **kargs):
        for key, value in kargs.items():
            self.values[key].update(value)
            
    def __getitem__(self, key):
        return self.values[key]
    
    def __setitem__(self, key, item):
        self.values[key].update(item)
        
class Checkpointer:
    def __init__(self, path, max_num=3):
        self.max_num = max_num
        self.path = path
        self.listfile = os.path.join(path, 'model_list.pkl')
        if not os.path.exists(self.listfile):
            with open(self.listfile, 'wb') as f:
                model_list = []
                pickle.dump(model_list, f)
        
    
    def save(self, model, optimizer, epoch):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        filename = os.path.join(self.path, 'model_{:05}.pth'.format(epoch))

        with open(self.listfile, 'rb+') as f:
            model_list = pickle.load(f)
            if len(model_list) >= self.max_num:
                os.remove(model_list[0])
                del model_list[0]
            model_list.append(filename)
        with open(self.listfile, 'rb+') as f:
            pickle.dump(model_list, f)
            
        with open(filename, 'wb') as f:
            torch.save(checkpoint, f)
    
    def load(self, model, optimizer):
        """
        Return starting epoch
        """
        with open(self.listfile, 'rb') as f:
            model_list = pickle.load(f)
            if len(model_list) == 0:
                print('No checkpoint found. Starting from scratch')
                return 0
            else:
                checkpoint = torch.load(model_list[-1])
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('Load checkpoint from {}.'.format(model_list[-1]))
                return checkpoint['epoch']
        

vis_logger = VisLogger()
metric_logger = MetricLogger()

