
import torch
import math
import numpy as np
from .importance import *

def expand_mask(mask, var_sizes):
    try:
        expanded_parts = []
        for i, size in enumerate(var_sizes):
            # Convert torch.Size to int (total number of elements)
            if isinstance(size, torch.Size):
                size_int = size.numel()  # Total number of elements
            else:
                size_int = size
            
            expanded_part = mask[:, i].view(mask.shape[0], 1).expand(mask.shape[0], size_int)
            expanded_parts.append(expanded_part)
        
        return torch.cat(expanded_parts, dim=1)
    except Exception as e:
        raise ValueError(f"mask: {mask}, var_sizes: {var_sizes}, error: {e}")


class VP_SDE():
    def __init__(self,
                 beta_min: float,
                 beta_max: float,
                 T: float,
                 importance_sampling: bool,
                 weight_s_functions: bool,
                 minde_type: str,
                 device: str = 'gpu'):
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.rand_batch = False
        self.T = T
        self.importance_sampling = importance_sampling
        self.weight_s_functions = weight_s_functions
        self.minde_type = minde_type
        self.device = device
        self.masks = self.get_masks_training()
       

    def set_device(self, device):
        self.device = device
        self.masks = self.masks.to(device)

    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, t):
        # Returns the drift and diffusion coefficient of the SDE ( f(t), g(t)) respectively.
        return -0.5*self.beta_t(t), torch.sqrt(self.beta_t(t))

    def marg_prob(self, t, x):
        
        ## Returns mean and std of the marginal distribution P_t(x_t) at time t.
        log_mean_coeff = -0.25 * t ** 2 * \
            (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        return mean.view(-1, 1) * torch.ones_like(x, device=self.device), std.view(-1, 1) * torch.ones_like(x, device=self.device)

    def sample(self, x_0, t):
        ## Forward SDE
        # Sample from P(x_t | x_0) at time t. Returns A noisy version of x_0.
        mean, std = self.marg_prob(t, t)
        z = torch.randn_like(x_0, device=self.device)
        x_t = x_0 * mean + std * z
        return x_t, z, mean, std

    def train_step(self, x, y, score_net, eps=1e-5, debug=True):
        """
        Perform a single training step for the SDE model.

        Args:
            data : The input data for the training step.
            score_net : The score network used for computing the score.
            eps: A small value used for numerical stability when importance sampling is Off. Defaults to 1e-5.
        Returns:
            Tensor: The loss value computed during the training step.
        """

        bs = x.shape[0]
        var_sizes = [x.shape[1:], y.shape[1:]]
        xy_0 = torch.cat((x, y), dim=1)

        if self.importance_sampling:
            t = (self.sample_importance_sampling_t(
                shape=(bs, 1))).to(self.device)
        else:
            t = ((self.T - eps) *
                 torch.rand((bs, 1)) + eps).to(self.device)
        # randomly sample an index to choose a masks
        if self.rand_batch:
            i = torch.randint(low=1, high=len(self.masks)+1, size=(bs,)) - 1
        else:
            i = (torch.randint(low=1, high=len(self.masks)+1, size=(1,)) - 1 ).expand(bs)
            
        # Select the mask randomly from the list of masks to learn the denoising score functions.

        if debug:
            # do normal diffusion model training
            xy_t, Z, mean, std = self.sample(x_0=xy_0, t=t)
            mask = self.masks[i, :]
            score = score_net(xy_t, t=t, mask=mask, std=None)
            x_denoised = (xy_t - std * score)/mean
            loss = torch.square(score - Z)
            # print(f"Norms: x0 {torch.norm(xy_0, p=2)}, xt {torch.norm(xy_t, p=2)}, z {torch.norm(Z, p=2)}, score {torch.norm(score, p=2)}, mean {torch.norm(mean, p=2)}, std {torch.norm(std, p=2)}, loss {torch.norm(loss, p=2)}")
            # print(f"mean Z {torch.mean(Z)}, mean score {torch.mean(score)}, mean loss {torch.mean(loss)}")
            # print(f"std Z {torch.std(std * score)}, std score {torch.std(score)}, std loss {torch.std(loss)}")
        else:

            mask = self.masks[i.long(), :]
            mask_data = expand_mask(mask, var_sizes)
            # Variables that are not marginal
            mask_data_marg = (mask_data < 0).float()
            # Variables that will be diffused
            mask_data_diffused = mask_data.clip(0, 1)
            xy_t, Z, mean, std = self.sample(x_0=xy_0, t=t)

            xy_t = mask_data_diffused * xy_t + (1 - mask_data_diffused) * xy_0
            xy_t = xy_t * (1 - mask_data_marg) + torch.zeros_like(xy_0, device=self.device) *mask_data_marg

            output = score_net(xy_t, t=t, mask=mask, std=None)
            score = output * mask_data_diffused

            Z = Z * mask_data_diffused

            x_denoised = (xy_t - std * score)/mean
            #Score matching of diffused data reweithed proportionnaly to the size of the diffused data.
            
            # Flatten the data
            score = score.view(bs, -1)
            mask_data_diffused = mask_data_diffused.view(bs, -1)
            Z = Z.view(bs, -1)

            total_size = score.size(1)
            n_diff=torch.sum(mask_data_diffused, dim=1)
            weight = (((total_size - n_diff) / total_size) + 1).view(bs, 1)
            loss = (weight * (torch.square(score - Z))).sum(1, keepdim=False)/n_diff

        ret_dict = {
            'loss': torch.mean(loss),
            'x_denoised': x_denoised,
            'score_norm': torch.norm(score, p=2),
            'noise_norm': torch.norm(Z, p=2),
            'score': score,
            'noise': std*Z,
            'xy_t': xy_t,
            't': t,
            'mask': mask,
            'mean': mean,
            'std': std
        }

        return loss, ret_dict

    
    def get_masks_training(self):
        """
        Returns a list of masks each corresponds to a score function needed to compute MI.
        
        
        """
        if self.minde_type=="c":
            masks= np.array([[1,-1],[1,0]]) 
        elif self.minde_type=="j":
            masks= np.array([[1,1],[1,0],[0,1]])  
        
        if self.weight_s_functions:
            return torch.tensor(self.weight_masks(masks), device=self.device)
        else:
            return  torch.tensor(masks, device=self.device)


    def weight_masks(self, masks):
        """ Weighting the mask list so the more complex score functions are picked more often durring the training step. 
        This is done by duplicating the mask with the list of masks.
        """
        masks_w = []
  
        #print("Weighting the scores to learn ")
        for s in masks:
                nb_var_inset = np.sum(s == 1)
                for i in range(nb_var_inset):
                    masks_w.append(s)
        np.random.shuffle(masks_w)
        return np.array(masks_w)
    
    def sample_importance_sampling_t(self, shape):
        """
        Non-uniform sampling of t to importance_sampling. See [1,2] for more details.
        [1] https://arxiv.org/abs/2106.02808
        [2] https://github.com/CW-Huang/sdeflow-light
        """
        return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, T=self.T)
