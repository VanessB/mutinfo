import math
import numpy
import torch

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_is_fitted, _is_fitted

from collections.abc import Callable


class AutoencoderPreprocessing(BaseEstimator, TransformerMixin):
    """
    Combination of transforms to be applied to elements of a tuple.
    """

    _parameter_constraints: dict = {
        "n_train_steps": [int],
        "train_batch_size": [int],
        "device": [str]
    }

    def __init__(
        self,
        autoencoder_factory: Callable[[tuple[int]], torch.nn.Module],
        n_train_steps: int=100000,
        train_batch_size: int=512,
        device: str="cpu"
    ) -> None:
        
        self.autoencoder_factory = autoencoder_factory
        self.n_train_steps = n_train_steps
        self.train_batch_size = train_batch_size
        self.device = device

        self._validate_params()

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        x, y = X

        x = torch.tensor(x, dtype=torch.float32).view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        y = torch.tensor(y, dtype=torch.float32).view(y.shape[0], -1, y.shape[-2], y.shape[-1])

        self.autoencoder = self.autoencoder_factory(x.shape).to(self.device)
        
        train_tensor = torch.cat([x, y], dim=0)
        train_dataset = torch.utils.data.TensorDataset(
            train_tensor,
            train_tensor
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=True,
        )

        self.autoencoder.train()
        loss = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1.0e-3)

        step = 0
        while step < self.n_train_steps:
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                x, y = batch
                loss(self.autoencoder(x.to(self.device)), y.to(self.device)).backward()

                optimizer.step()
                step += 1

        return self

    def transform(self, X) -> tuple:
        check_is_fitted(self)

        x, y = X

        x = torch.tensor(x, dtype=torch.float32).view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        y = torch.tensor(y, dtype=torch.float32).view(y.shape[0], -1, y.shape[-2], y.shape[-1])

        eval_dataset = torch.utils.data.TensorDataset(x, y)
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.train_batch_size,
            shuffle=False,
            pin_memory=True,
        )

        all_x = []
        all_y = []
        with torch.no_grad():
            self.autoencoder.eval()
            for batch in eval_dataloader:
                x, y = batch
                
                all_x.append(self.autoencoder.encoder(x.to(self.device)).cpu().numpy())
                all_y.append(self.autoencoder.encoder(y.to(self.device)).cpu().numpy())

        return (numpy.concatenate(all_x, axis=0), numpy.concatenate(all_y, axis=0))

    def __sklearn_is_fitted__(self) -> bool:
        return True


class GenericAutoencoderConv2d(torch.nn.Module):
    """
    A very simple convolutional autoencoder.
    """
    
    def __init__(self, image_shape: tuple[int], hidden_dim: int=2, leaky=0.2) -> None:
        super().__init__()

        if (not len(image_shape) in [3, 4]):
            raise ValueError("Inputs shpuld be batches of images.")

        class DownConvBlock(torch.nn.Module):
            def __init__(self, in_channels: int, out_channels: int, leaky: float=0.2) -> None:
                super().__init__()

                self.convolution = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
                self.pooling     = torch.nn.MaxPool2d(2)
                self.batchnorm   = torch.nn.BatchNorm2d(out_channels)
                self.activation  = torch.nn.LeakyReLU(leaky)

            def forward(self, x):
                x = self.convolution(x)
                x = self.pooling(x)
                x = self.batchnorm(x)
                x = self.activation(x)

                return x

        class UpConvBlock(torch.nn.Module):
            def __init__(self, in_channels: int, out_channels: int, leaky: float=0.2) -> None:
                super().__init__()

                self.convolution = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
                self.upsample    = torch.nn.Upsample(scale_factor=2)
                self.batchnorm   = torch.nn.BatchNorm2d(out_channels)
                self.activation  = torch.nn.LeakyReLU(leaky)

            def forward(self, x):
                x = self.upsample(x)
                x = self.convolution(x)
                x = self.batchnorm(x)
                x = self.activation(x)

                return x

        image_power = int(math.floor(math.log2(image_shape[-1])))
        in_channels = image_shape[1] if (len(image_shape) == 4) else 1
        
        self.encoder = torch.nn.Sequential(
            DownConvBlock(in_channels, 4, leaky),
            DownConvBlock(4, 8, leaky),
            *[DownConvBlock(8, 8, leaky) for _ in range(2, image_power)],
            torch.nn.Flatten(),
            torch.nn.Linear(8, hidden_dim),
            torch.nn.Tanh(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 8),
            torch.nn.LeakyReLU(leaky),
            torch.nn.Unflatten(dim=-1, unflattened_size=(8, 1, 1)),
            *[UpConvBlock(8, 8, leaky) for _ in range(1, image_power)],
            UpConvBlock(8, 4, leaky),
            torch.nn.Conv2d(4, 1, kernel_size=3, padding="same"),
        )

    def forward(self, z):
        return self.decoder(self.encoder(z))