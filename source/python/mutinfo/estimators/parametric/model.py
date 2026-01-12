import torch
from copy import deepcopy
from diffusers import UNet2DModel
import pytorch_lightning as pl
import math
import numpy as np

class EMA(torch.nn.Module):
    """ Model Exponential Moving Average V2 from timm"""
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class GenericConv2DDiffuser(torch.nn.Module):

    def __init__(
        self,
        X_shape: tuple,
        Y_shape: tuple,
        n_filters: int=16,
        hidden_dim: int=128,
        n_X_convolutions: int=None,
        n_Y_convolutions: int=None,
    ) -> None:
        super().__init__()

        if (not len(X_shape) in [3, 4]) or (not len(Y_shape) in [3, 4]):
            raise ValueError("Inputs shpuld be batches of images.")

        if (X_shape[-2] != X_shape[-1]) or (Y_shape[-2] != Y_shape[-1]):
            raise ValueError("Input images have to be square.")

        n_X_channels = X_shape[1] if (len(X_shape) == 4) else 1
        n_Y_channels = Y_shape[1] if (len(Y_shape) == 4) else 1

        self.n_X_channels = n_X_channels
        self.n_Y_channels = n_Y_channels

        self.total_X_size = np.prod(X_shape)
        self.total_Y_size = np.prod(Y_shape)

        self.X_shape = X_shape
        self.Y_shape = Y_shape

        self.n_filters = n_filters

        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(2, 2*n_filters),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2*n_filters, 2*n_filters)
        )
        
        # Convolution layers.
        # TODO: reuse code!
        if n_X_convolutions is None:
            log2_remaining_size = 2
            n_X_convolutions = int(math.floor(math.log2(X_shape[-1]))) - log2_remaining_size
            
        self.X_convolutions = torch.nn.ModuleList([torch.nn.Conv2d(n_X_channels, n_filters, kernel_size=3, padding='same')] + \
                [torch.nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='same') for index in range(n_X_convolutions - 1)])
        for conv_index in range(n_X_convolutions):
            X_shape = X_shape[:-2] + ((X_shape[-2] - 2) // 2 + 1, (X_shape[-1] - 2) // 2 + 1,)

        if n_Y_convolutions is None:
            log2_remaining_size = 2
            n_Y_convolutions = int(math.floor(math.log2(Y_shape[-1]))) - log2_remaining_size
        for conv_index in range(n_Y_convolutions):
            Y_shape = Y_shape[:-2] + ((Y_shape[-2] - 2) // 2 + 1, (Y_shape[-1] - 2) // 2 + 1,)
            
        self.Y_convolutions = torch.nn.ModuleList([torch.nn.Conv2d(n_Y_channels, n_filters, kernel_size=3, padding='same')] + \
                [torch.nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='same') for index in range(n_Y_convolutions - 1)])

        self.activation = torch.nn.LeakyReLU()
        self.maxpool2d = torch.nn.MaxPool2d((2,2))

        # Dense part.
        remaining_dim_X = n_filters * X_shape[-1] * X_shape[-2]
        remaining_dim_Y = n_filters * Y_shape[-1] * Y_shape[-2]
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(remaining_dim_X + remaining_dim_Y, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, self.total_X_size + self.total_Y_size),
        )

    def forward(self, xy: torch.Tensor, cond: torch.Tensor, std=None) -> torch.tensor:
        xy_shape = xy.shape
        try:
            x = xy[:, :self.total_X_size].reshape(-1, *self.X_shape)
        except:
            raise UserWarning(f"Shape mismatch xy: {xy.shape}, total_X_size: {self.total_X_size}, X_shape: {self.X_shape}")
        y = xy[:, self.total_X_size:].reshape(-1, *self.Y_shape)

        assert x.shape[0] == y.shape[0], f"Batch size mismatch x: {x.shape}, y: {y.shape}, {self.total_X_size}, xy_shape: {xy_shape}, X_shape: {self.X_shape}, Y_shape: {self.Y_shape}"

        t_xy = self.time_mlp(cond)

        t_x = t_xy[:, :self.n_filters].reshape(-1, self.n_filters, 1, 1)
        t_y = t_xy[:, self.n_filters:].reshape(-1, self.n_filters, 1, 1)

        # Convolution layers.
        for conv2d in self.X_convolutions:
            x = conv2d(x)
            try:
                x = x + t_x
            except:
                raise ValueError(f"Shapes - x: {x.shape}, y: {y.shape}, t_x: {t_x.shape}, conv2d: {conv2d}, X_shape: {self.X_shape}, Y_shape: {self.Y_shape}, total_X_size: {self.total_X_size}, total_Y_size: {self.total_Y_size}")
            x = self.maxpool2d(x)
            x = self.activation(x)
            
        for conv2d in self.Y_convolutions:
            y = conv2d(y)
            y = y + t_y
            y = self.maxpool2d(y)
            y = self.activation(y)

        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        
        xy = self.dense(torch.cat((x, y), dim=1))

        if std != None:
            xy = xy / std
        return xy

class GenericMLPDiffuser(torch.nn.Module):

    def __init__(
        self,
        X_shape: tuple,
        Y_shape: tuple,
        hidden_dim: int=128,
    ) -> None:
        super().__init__()
        self.total_X_size = np.prod(X_shape)
        self.total_Y_size = np.prod(Y_shape)
        self.linear1 = torch.nn.Linear(self.total_X_size + self.total_Y_size, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, self.total_X_size + self.total_Y_size)
        self.act = torch.nn.LeakyReLU()

        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, xy: torch.Tensor, cond: torch.Tensor, std=None) -> torch.tensor:
        t_xy = self.time_mlp(cond)
        xy = self.linear1(xy)
        xy = xy + t_xy
        xy = self.act(xy)
        xy = self.linear2(xy)
        xy = xy + t_xy
        xy = self.act(xy)
        xy = self.linear3(xy)
        if std is not None:
            xy = xy / std
        return xy

# UNet backbone for diffusion
class UNet(torch.nn.Module):
    def __init__(self, X_shape, Y_shape, hidden_dim=64, nb_var=2, norm_num_groups=8, name="mod", layers_per_block=2):
        super(UNet, self).__init__()
        self.name =name
        self.X_shape = X_shape
        self.Y_shape = Y_shape
        self.total_X_size = X_shape[-1] * X_shape[-2]
        self.total_Y_size = Y_shape[-1] * Y_shape[-2]
        # UNet from diffusers library
        sample_size = X_shape[-1]
        channels_x = X_shape[0]
        channels_y = Y_shape[0]
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=channels_x + channels_y,
            out_channels=channels_x + channels_y,
            layers_per_block=layers_per_block,
            block_out_channels=(hidden_dim, hidden_dim * 2, hidden_dim * 4),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
            norm_num_groups=norm_num_groups,
        )
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(nb_var, 1),
            torch.nn.GELU(),
            torch.nn.Linear(1, 1)
        )


    def forward(self, x, timestep, std=None):
        # Pass the latent through the U-Net for denoising
        x = x.reshape(-1, self.X_shape[0]+self.Y_shape[0], *self.X_shape[-2:])
        timestep = self.time_mlp(timestep).squeeze(-1)
        ret = self.unet(x, timestep)["sample"]
        return ret.reshape(ret.shape[0], -1)