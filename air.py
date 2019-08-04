from torch import nn
import torch
from torch.nn import LSTMCell
from attrdict import AttrDict
import torch.nn.functional as F
from collections import namedtuple
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from copy import deepcopy
from matplotlib import pyplot as plt

# default air architecture
default_arch = AttrDict({
    'max_steps': 3,
    
    # network related
    'input_shape': (50, 50),
    'object_shape': (20, 20),
    'input_size': 50 * 50,
    'object_size': 20 * 20,
    'z_what_size': 50,
    'lstm_hidden_size': 256,
    'baseline_hidden_size': 256,
    # encode object into z_what
    'encoder_hidden_size': 200,
    'decoder_hidden_size': 200,
    
    # priors
    'z_pres_prob_prior': 0.5,
    'z_where_loc_prior': torch.tensor([3.0, 0.0, 0.0]),
    'z_where_scale_prior': torch.tensor([0.2, 1.0, 1.0]),
    'z_what_loc_prior': torch.tensor(0.0),
    'z_what_scale_prior': torch.tensor(1.0),
    
    # output prior
    'x_scale': torch.tensor(0.3)
})

class SpatialTransformer(nn.Module):
    def __init__(self, input_size, output_size):
        """
        :param input_size: (H, W)
        :param output_size: (H, W)
        """
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        
    def forward(self, x, z_where, inverse=False):
        """
        :param x: (B, 1, Hin, Win)
        :param z_where: [s, x, y]
        :param inverse: inverse z_where
        :return: y of output_size
        """
        B = x.size(0)
        theta = self.z_where_to_matrix(z_where, inverse)
        grid = F.affine_grid(theta, torch.Size((B, 1) + (self.output_size)))
        out = F.grid_sample(x, grid)
        return out
        
    @staticmethod
    def z_where_to_matrix(z_where, inverse=False):
        """
        :param z_where: batch. [s, x, y]
        :param inverse: transform [s, x, y] to [1/s, -x/s, -y/s]
        :return: [[s, 0, x], [0, s, y]]
        """
        B = z_where.size(0)
        
        if inverse:
            z_where_inv = z_where.clone()
            z_where_inv[:, 1:3] = -z_where[:, 1:3] / z_where[:, 0:1]
            z_where_inv[:, 0:1] = 1 / z_where[:, 0:1]
            z_where = z_where_inv
            
        # [0, s, x, y] -> [s, 0, x, 0, s, y]
        z_where = torch.cat((torch.zeros(B, 1), z_where), dim=1)
        expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])
        matrix = torch.index_select(z_where, dim=1, index=expansion_indices)
        matrix = matrix.view(B, 2, 3)
        
        return matrix
        
        
    
class AIRState(AttrDict):
    def __init__(self, z_pres, z_where, z_what, h, c, bl_c, bl_h):
        """
        Note that z_where is for image to object transformation
        """
        AttrDict.__init__(self,
            z_pres=z_pres,
            z_where=z_where,
            z_what=z_what,
            h=h, c=c, bl_h=bl_h, bl_c=bl_c)
        
    @staticmethod
    def get_intial_state(B, arch):
        """
        :param B: batch size
        """
        return AIRState(
            z_pres=torch.ones(B, 1),
            z_where=torch.zeros(B, 3),
            z_what=torch.zeros(B, arch.z_what_size),
            h=torch.zeros(B, arch.lstm_hidden_size),
            c=torch.zeros(B, arch.lstm_hidden_size),
            bl_c=torch.zeros(B, arch.baseline_hidden_size),
            bl_h=torch.zeros(B, arch.baseline_hidden_size),
        )

class Predict(nn.Module):
    """
    Given h that encodes z[1:i-1] and x, predict z_pres and z_where
    """
    def __init__(self, arch):
        nn.Module.__init__(self)
        self.fc = nn.Linear(arch.lstm_hidden_size, 7)
        
    def forward(self, h):
        z = self.fc(h)
        z_pres_p = torch.sigmoid(z[:, :1])
        z_where_loc = z[:, 1:4]
        z_where_scale = F.softplus(z[:, 4:])
        
        return z_pres_p, z_where_loc, z_where_scale
    
    
class Encoder(nn.Module):
    """
    Given crop object, predict z_what
    """
    def __init__(self, arch):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(arch.object_size, arch.encoder_hidden_size)
        self.fc2 = nn.Linear(arch.encoder_hidden_size, arch.z_what_size * 2)
        self.what_size = arch.z_what_size

    def forward(self, object):
        """
        :param object: (B, 1, H, W)
        :return: z_what_loc, z_what_scale
        """
        B = object.size(0)
        object_flat = object.view(B, -1)
        x = F.relu(self.fc1(object_flat))
        x = self.fc2(x)
        z_what_loc, z_what_scale = x[:, :self.what_size], x[:, self.what_size:]
        z_what_scale = F.softplus(z_what_scale)
        
        return z_what_loc, z_what_scale
    
class Decoder(nn.Module):
    """
    Given z_what, decoder it into object
    """
    def __init__(self, arch):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(arch.z_what_size, arch.decoder_hidden_size)
        self.fc2 = nn.Linear(arch.decoder_hidden_size, arch.input_size)
        self.h, self.w = arch.input_shape
        
    def forward(self, z_what):
        x = F.relu(self.fc1(z_what))
        x = self.fc2(x)
        x = x.view(-1, 1, self.h, self.w)
        x = torch.sigmoid(x - 2)
        return x
    


class AIR(nn.Module):
    counter = 0
    
    def __init__(self, arch=None):
        """
        :param arch: dictionary, for overriding default architecture
        """
        nn.Module.__init__(self)
        self.arch = deepcopy(default_arch)
        if arch is not None:
            self.arch.update(arch)
            
            
        self.T = self.arch.max_steps
        
        # 4: where + pres
        lstm_input_size = self.arch.input_size + self.arch.z_what_size + 4
        self.lstm_cell = LSTMCell(lstm_input_size, self.arch.lstm_hidden_size)
        
        # predict z_where, z_pres from h
        self.predict = Predict(self.arch)
        # encode object into what
        self.encoder = Encoder(self.arch)
        # decode what into object
        self.decoder = Decoder(self.arch)
        
        # spatial transformers
        self.image_to_object = SpatialTransformer(self.arch.input_shape, self.arch.object_shape)
        self.object_to_image = SpatialTransformer(self.arch.object_shape, self.arch.input_shape)

        # baseline RNN
        self.bl_rnn = LSTMCell(lstm_input_size, self.arch.baseline_hidden_size)
        # predict baseline value
        self.bl_predict = nn.Linear(self.arch.baseline_hidden_size, 1)
        
        # priors
        self.pres_prior = Bernoulli(probs=self.arch.z_pres_prob_prior)
        self.where_prior = Normal(loc=self.arch.z_where_loc_prior, scale=self.arch.z_where_scale_prior)
        self.what_prior = Normal(loc=self.arch.z_what_loc_prior, scale=self.arch.z_what_scale_prior)
        
        # modules excluding baseline rnn
        self.air_modules = nn.ModuleList([
            self.predict, self.lstm_cell, self.encoder, self.decoder
        ])

        self.baseline_modules = nn.ModuleList([
            self.bl_rnn,
            self.bl_predict
        ])
        
    def forward(self, x):
        B = x.size(0)
        state = AIRState.get_intial_state(B, self.arch)

        # accumulated KL divergence
        kl = []
        # baseline value for each step
        baseline_value = []
        # z_pres likelihood for each step
        z_pres_likelihood = []
        # learning signal for each step
        learning_signal = torch.zeros(B, self.arch.max_steps)
        # mask for nonexistent ones
        mask = torch.ones(B, self.arch.max_steps)
        # canvas
        h, w = self.arch.input_shape
        canvas = torch.zeros(B, 1, h, w)
        
        for t in range(self.T):
            state, this_kl, this_baseline_value, this_z_pres_likelihood = self.infer_step(state, x)
            baseline_value.append(this_baseline_value.squeeze())
            kl.append(this_kl)
            z_pres_likelihood.append(this_z_pres_likelihood.squeeze())
            for j in range(t):
                learning_signal[:, j] += this_kl.squeeze()
                
            # reconstruct
            object = self.decoder(state.z_what)
            # (B, 1, H, W)
            img = self.object_to_image(object, state.z_where, inverse=False)
            canvas = canvas + img * state.z_pres[:, :, None, None]
            
            mask[:, t] = state.z_pres.squeeze()
            
        self.counter += 1
        if self.counter % 20 == 0:
            plt.subplot(2, 1, 1)
            plt.imshow(canvas[0][0].cpu().detach().numpy())
            plt.subplot(2, 1, 2)
            plt.imshow(x[0][0].cpu().detach().numpy())
            number = mask[0].sum()
            plt.title('pred: {}'.format(number))
            plt.savefig('logs/{}.png'.format(self.counter))
            
        baseline_value = torch.stack(baseline_value, dim=1)
        kl = torch.stack(kl, dim=1)
        z_pres_likelihood = torch.stack(z_pres_likelihood, dim=1)
        
        # construct output distribution
        output_dist = Normal(canvas, self.arch.x_scale.expand(canvas.shape))
        likelihood = output_dist.log_prob(x)
        # sum over data dimension
        likelihood = likelihood.view(B, -1).sum(1)
        
        # construct surrogate loss
        # reinforce term
        learning_signal = learning_signal + likelihood[:, None]
        learning_signal = learning_signal * mask
        reinforce_term = (learning_signal.detach() - baseline_value.detach())* z_pres_likelihood
        reinforce_term = reinforce_term.sum(1)
        
        # kl term, sum over batch dimension
        kl = kl.sum(1)
        
        loss = reinforce_term + kl - likelihood
        # mean over batch dimension
        loss = loss.mean()
        
        
        # compute baseline loss
        baseline_loss = F.mse_loss(baseline_value, learning_signal.detach())

        losslist = (reinforce_term.mean(), kl.mean(), likelihood.mean(), baseline_loss)
        print(*[x.item() for x in losslist])
        
        return loss + baseline_loss
        

    def infer_step(self, prev, x):
        """
        Given previous state, predict next state. We assume that z_pres is 1
        :param prev: AIRState
        :return: new_state, KL, baseline value, z_pres_likelihood
        """
        
        B = x.size(0)
        
        # Flatten x
        x_flat = x.view(B, -1)
        
        # First, compute h_t that encodes (x, z[1:i-1])
        lstm_input = torch.cat((x_flat, prev.z_where, prev.z_what, prev.z_pres), dim=1)
        h, c = self.lstm_cell(lstm_input, (prev.h, prev.c))
        
        # Predict presence and location
        z_pres_p, z_where_loc, z_where_scale = self.predict(h)
        
        # In theory, if z_pres is 0, we don't need to continue computation. But
        # for batch processing, we will do this anyway.
        
        # sample z_pres
        z_pres_post = Bernoulli(z_pres_p)
        z_pres = z_pres_post.sample()
        z_pres = z_pres * prev.z_pres
        # likelihood
        z_pres_likelihood = z_pres_post.log_prob(z_pres) * z_pres
        # (B,)
        z_pres_likelihood = z_pres_likelihood.squeeze()
        
        
        # sample z_where
        z_where_post = Normal(z_where_loc, z_where_scale)
        z_where = z_where_post.rsample()
        
        # extract object
        # (B, 1, Hobj, Wobj)
        object = self.image_to_object(x, z_where, inverse=True)
        
        # predict z_what
        z_what_loc, z_what_scale = self.encoder(object)
        z_what_post = Normal(z_what_loc, z_what_scale)
        z_what = z_what_post.rsample()
        # z_what *= prev.z_pres[:, None]
        
        
        # compute baseline for this z_pres
        bl_h, bl_c = self.bl_rnn(lstm_input.detach(), (prev.bl_h, prev.bl_c))
        # (B,)
        baseline_value = self.bl_predict(bl_h).squeeze()
        baseline_value = baseline_value * z_pres.squeeze()
        
        # compute KL as we go, sum over data dimension
        kl_pres = kl_divergence(z_pres_post, self.pres_prior.expand(z_pres_post.batch_shape)).sum(1)
        kl_where = kl_divergence(z_where_post, self.where_prior.expand(z_where_post.batch_shape)).sum(1)
        kl_what = kl_divergence(z_what_post, self.what_prior.expand(z_what_post.batch_shape)).sum(1)
        
        # mask out non-existent terms
        kl = (kl_pres + kl_where + kl_what) * z_pres.squeeze()

        # new state
        new_state = AIRState(z_pres=z_pres, z_where=z_where, z_what=z_what,
                             h=h, c=c, bl_c=bl_c, bl_h=bl_h)
        
        return new_state, kl, baseline_value, z_pres_likelihood
    
if __name__ == '__main__':
    model = AIR()
    img = torch.rand(4, 1, 50, 50)
    loss = model(img)
