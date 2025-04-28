import math
import numpy
import torch
import torchkld

from collections.abc import Callable

from sklearn.model_selection import train_test_split

from ..base import MutualInformationEstimator

_EPS = 1.0e-6


class _MINE_backbone(torchkld.mutual_information.MINE):
    def __init__(self, network: torch.nn.Module, concatenate: bool=True) -> None:
        super().__init__()

        self.network = network
        self.concatenate = concatenate

    @torchkld.mutual_information.MINE.marginalizable
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.network(torch.concat([x, y], dim=1)) if self.concatenate else self.network(x, y)


class MINE(MutualInformationEstimator):
    def __init__(
        self,
        backbone_factory: Callable[[], torchkld.mutual_information.MINE]=None,
        loss_factory: Callable[ [], Callable[[torch.Tensor, torch.Tensor], torch.Tensor] ]=None,
        optimizer_factory: Callable[[], torch.optim.Optimizer]=None,
        #n_train_epochs: int=100,
        n_train_steps: int=10000,
        train_batch_size: int=512,
        estimate_batch_size: int=512,
        estimate_fraction: float=0.5,
        marginalize: str="permute",
        device: str="cpu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.backbone_factory = backbone_factory
        if self.backbone_factory is None:
            self.backbone_factory = GenericMLPClassifier

        self.loss_factory = loss_factory
        if self.loss_factory is None:
            self.loss_factory = lambda : torchkld.loss.DonskerVaradhanLoss(ema_multiplier=1.0e-2)

        self.optimizer_factory = optimizer_factory
        if self.optimizer_factory is None:
            self.optimizer_factory = lambda parameters : torch.optim.Adam(parameters, lr=1.0e-3)

        self.n_train_steps = n_train_steps
        self.train_batch_size = train_batch_size
        self.estimate_batch_size = estimate_batch_size
        self.estimate_fraction = estimate_fraction
        self.marginalize = marginalize
        self.device = device

    def __call__(self, x: numpy.ndarray, y: numpy.ndarray) -> float:
        """
        Estimate the value of mutual information between two random vectors
        using samples `x` and `y`.

        Parameters
        ----------
        x, y : array_like
            Samples from corresponding random vectors.

        Returns
        -------
        mutual_information : float
            Estimated value of mutual information.
        """

        self._check_arguments(x, y)

        if self.estimate_fraction is None:
            train_x, estimate_x, train_y, estimate_y = x, x, y, y
        else:
            train_x, estimate_x, train_y, estimate_y = train_test_split(x, y, test_size=self.estimate_fraction)
            
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_x, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
        )

        estimate_dataset = torch.utils.data.TensorDataset(
            torch.tensor(estimate_x, dtype=torch.float32),
            torch.tensor(estimate_y, dtype=torch.float32),
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=True,
        )

        estimate_dataloader = torch.utils.data.DataLoader(
            estimate_dataset,
            batch_size=self.estimate_batch_size,
            shuffle=False,
            pin_memory=True,
        )

        backbone = self.backbone_factory(x.shape, y.shape).to(self.device)
        loss = self.loss_factory()
        optimizer = self.optimizer_factory(backbone.parameters())

        step = 0
        while step < self.n_train_steps:
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                x, y = batch
                loss(
                    backbone(x.to(self.device), y.to(self.device)),
                    backbone(x.to(self.device), y.to(self.device), marginalize=self.marginalize),
                ).backward()

                optimizer.step()
                step += 1

        estimated_MI = backbone.get_mutual_information(
            estimate_dataloader,
            loss,
            self.device,
            marginalize=self.marginalize
        )

        return max(estimated_MI, 0.0)


def GenericMLPClassifier(
    X_shape: tuple,
    Y_shape: tuple,
    hidden_dim: int=128
) -> _MINE_backbone:
    return _MINE_backbone(
        torch.nn.Sequential(
            torch.nn.Linear(X_shape[-1] + Y_shape[-1], hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
    )

class GenericConv2dClassifier(torchkld.mutual_information.MINE):
    def __init__(
        self,
        X_shape: tuple,
        Y_shape: tuple,
        n_filters: int=16,
        hidden_dim: int=128
    ) -> None:
        super().__init__()

        if (not len(X_shape) in [3, 4]) or (not len(Y_shape) in [3, 4]):
            raise ValueError("Inputs shpuld be batches of images.")

        if (X_shape[-2] != X_shape[-1]) or (Y_shape[-2] != Y_shape[-1]):
            raise ValueError("Input images have to be square.")

        n_X_channels = X_shape[1] if (len(X_shape) == 4) else 1
        n_Y_channels = Y_shape[1] if (len(Y_shape) == 4) else 1
        log2_remaining_size = 2
        
        # Convolution layers.
        n_X_convolutions = int(math.floor(math.log2(X_shape[-1]))) - log2_remaining_size
        self.X_convolutions = torch.nn.ModuleList([torch.nn.Conv2d(n_X_channels, n_filters, kernel_size=3, padding='same')] + \
                [torch.nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='same') for index in range(n_X_convolutions - 1)])
            
        n_Y_convolutions = int(math.floor(math.log2(Y_shape[-1]))) - log2_remaining_size
        self.Y_convolutions = torch.nn.ModuleList([torch.nn.Conv2d(n_Y_channels, n_filters, kernel_size=3, padding='same')] + \
                [torch.nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='same') for index in range(n_Y_convolutions - 1)])

        self.activation = torch.nn.LeakyReLU()
        self.maxpool2d = torch.nn.MaxPool2d((2,2))

        # Dense part.
        remaining_dim = n_filters * 2**(2*log2_remaining_size)
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(remaining_dim + remaining_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    @torchkld.mutual_information.MINE.marginalizable
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.tensor:
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        y = y.view(y.shape[0], -1, y.shape[-2], y.shape[-1])
            
        # Convolution layers.
        for conv2d in self.X_convolutions:
            x = conv2d(x)
            x = self.maxpool2d(x)
            x = self.activation(x)
            
        for conv2d in self.Y_convolutions:
            y = conv2d(y)
            y = self.maxpool2d(y)
            y = self.activation(y)

        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        
        return self.dense(torch.cat((x, y), dim=1))