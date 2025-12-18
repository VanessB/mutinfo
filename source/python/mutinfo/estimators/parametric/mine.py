import math
import numpy
import torch
import torchfd

from collections.abc import Callable

from sklearn.model_selection import train_test_split

from ..base import MutualInformationEstimator

_EPS = 1.0e-6


class _MINE_backbone(torchfd.mutual_information.MINE):
    def __init__(
        self,
        network: torch.nn.Module,
        concatenate: bool=True,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.network = network
        self.concatenate = concatenate

    @torchfd.mutual_information.MINE.marginalized
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.network(torch.concat([x, y], dim=1)) if self.concatenate else self.network(x, y)


class MINE(MutualInformationEstimator):
    def __init__(
        self,
        backbone_factory: Callable[[], torchfd.mutual_information.MINE]=None,
        marginalizer_factory: Callable[[], torchfd.mutual_information.Marginalizer]=None,
        loss_factory: Callable[ [], Callable[[torch.Tensor, torch.Tensor], torch.Tensor] ]=None,
        optimizer_factory: Callable[[], torch.optim.Optimizer]=None,
        n_train_steps: int=10000,
        train_batch_size: int=512,
        estimate_batch_size: int=512,
        estimate_size: float=0.5,
        clip: float=None,
        device: str="cpu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.backbone_factory = backbone_factory
        if self.backbone_factory is None:
            self.backbone_factory = GenericMLPClassifier

        self.marginalizer_factory = marginalizer_factory
        if self.marginalizer_factory is None:
            self.marginalizer_factory = torchfd.mutual_information.PermutationMarginalizer

        self.loss_factory = loss_factory
        if self.loss_factory is None:
            self.loss_factory = lambda : torchfd.loss.DonskerVaradhanLoss(ema_multiplier=1.0e-2)

        self.optimizer_factory = optimizer_factory
        if self.optimizer_factory is None:
            self.optimizer_factory = lambda parameters : torch.optim.Adam(parameters, lr=1.0e-3)

        self.n_train_steps = n_train_steps
        self.train_batch_size = train_batch_size
        self.estimate_batch_size = estimate_batch_size
        self.estimate_size = estimate_size
        self.clip = clip
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

        if self.estimate_size is None:
            train_x, estimate_x, train_y, estimate_y = x, x, y, y
        else:
            train_x, estimate_x, train_y, estimate_y = train_test_split(x, y, test_size=self.estimate_size)
            
        
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
            pin_memory=False,
        )

        estimate_dataloader = torch.utils.data.DataLoader(
            estimate_dataset,
            batch_size=self.estimate_batch_size,
            shuffle=False,
            pin_memory=False,
        )

        backbone = self.backbone_factory(
            x.shape,
            y.shape,
            marginalizer=self.marginalizer_factory()
        ).to(self.device)
        loss = self.loss_factory()
        optimizer = self.optimizer_factory(backbone.parameters())

        step = 0
        while step < self.n_train_steps:
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                x, y = batch
                loss(*backbone(x.to(self.device), y.to(self.device))).backward()

                optimizer.step()
                step += 1

        estimated_MI = backbone.get_mutual_information(
            estimate_dataloader,
            loss,
            self.device,
            clip=self.clip,
        )

        return max(estimated_MI, 0.0)


def GenericMLPClassifier(
    X_shape: tuple,
    Y_shape: tuple,
    hidden_dim: int=128,
    *args, **kwargs
) -> _MINE_backbone:
    return _MINE_backbone(
        torch.nn.Sequential(
            torch.nn.Linear(X_shape[-1] + Y_shape[-1], hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, 1),
        ),
        *args, **kwargs
    )

class GenericConv2dClassifier(torchfd.mutual_information.MINE):
    def __init__(
        self,
        X_shape: tuple,
        Y_shape: tuple,
        n_filters: int=16,
        hidden_dim: int=16,
        n_X_convolutions: int=None,
        n_Y_convolutions: int=None,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        if (not len(X_shape) in [3, 4]) or (not len(Y_shape) in [3, 4]):
            raise ValueError("Inputs shpuld be batches of images, instead got input shapes {} and {}".format(X_shape, Y_shape))
            
        self.X_convolutions, X_final_shape = self.build_conv2d_tower(X_shape, n_filters)
        self.Y_convolutions, Y_final_shape = self.build_conv2d_tower(Y_shape, n_filters)

        self.activation = torch.nn.LeakyReLU()
        self.maxpool2d = torch.nn.MaxPool2d((2,2))

        # Dense part.
        X_remaining_dim = n_filters * X_final_shape[-1] * X_final_shape[-2]
        Y_remaining_dim = n_filters * Y_final_shape[-1] * Y_final_shape[-2]
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(X_remaining_dim + Y_remaining_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def build_conv2d_tower(
        self,
        shape: tuple[int],
        n_filters: int,
        conv2d_params: dict={"kernel_size": 3, "padding": 'same'},
        n_convolutions: int=None,
    ) -> tuple[torch.nn.ModuleList, tuple[int]]:
        if len(shape) == 3:
            shape = (shape[0], 1, shape[1], shape[2])
            
        n_channels = shape[1]
        min_size = min(shape[2], shape[3])

        if n_convolutions is None:
            log2_remaining_size = 2
            n_convolutions = int(math.floor(math.log2(min_size))) - log2_remaining_size
            
        convolutions = torch.nn.ModuleList([torch.nn.Conv2d(n_channels, n_filters, **conv2d_params)] + \
                [torch.nn.Conv2d(n_filters, n_filters, **conv2d_params) for index in range(n_convolutions - 1)])
        for conv_index in range(n_convolutions):
            shape = shape[:-2] + ((shape[-2] - 2) // 2 + 1, (shape[-1] - 2) // 2 + 1,)

        return convolutions, shape

    @torchfd.mutual_information.MINE.marginalized
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