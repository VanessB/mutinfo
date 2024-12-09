import numpy
import torch
import torchkld

from collections.abc import Callable

from ..base import MutualInformationEstimator


class _MINE_backbone(torchkld.mutual_information.MINE):
    def __init__(self, network: torch.nn.Module, concatenate: bool=True) -> None:
        super().__init__()

        self.network = network
        self.concatenate = concatenate

    def forward(self, x: torch.Tensor, y: torch.Tensor, marginalize: bool=False) -> torch.Tensor:
        x, y = super().forward(x, y, marginalize)

        return self.network(torch.concat([x, y], dim=1)) if self.concatenate else self.network(x, y)


class MINE(MutualInformationEstimator):
    def __init__(
        self,
        backbone_factory: Callable[[], torchkld.mutual_information.MINE]=None,
        loss_factory: Callable[ [], Callable[[torch.Tensor, torch.Tensor], torch.Tensor] ]=None,
        optimizer_factory: Callable[[], torch.optim.Optimizer]=None,
        n_train_epochs: int=100,
        train_batch_size: int=1024,
        estimate_batch_size: int=1024,
        marginalize: str="permute",
        device: str="cpu"
    ) -> None:
        super().__init__()

        self.backbone_factory = backbone_factory
        if backbone_factory is None:
            self.backbone_factory = lambda X_shape, Y_shape : _MINE_backbone(
                torch.nn.Sequential(
                    torch.nn.Linear(X_shape[-1] + Y_shape[-1], 128),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(128, 128),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(128, 1),
                )
            )

        self.loss_factory = loss_factory
        if self.loss_factory is None:
            self.loss_factory = lambda : torchkld.loss.NWJLoss()

        self.optimizer_factory = optimizer_factory
        if self.optimizer_factory is None:
            self.optimizer_factory = lambda parameters : torch.optim.Adam(parameters, lr=5.0e-3)

        self.n_train_epochs = n_train_epochs
        self.train_batch_size = train_batch_size
        self.estimate_batch_size = estimate_batch_size
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
        std : bool
            Calculate standard deviation.

        Returns
        -------
        mutual_information : float
            Estimated value of mutual information.
        """

        self._check_arguments(x, y)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
        )

        estimate_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.estimate_batch_size,
            shuffle=False,
        )

        backbone = self.backbone_factory(x.shape, y.shape).to(self.device)
        loss = self.loss_factory()
        optimizer = self.optimizer_factory(backbone.parameters())
        
        for epoch in range(self.n_train_epochs):
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                x, y = batch
                loss(
                    backbone(x.to(self.device), y.to(self.device)),
                    backbone(x.to(self.device), y.to(self.device), marginalize=self.marginalize),
                ).backward()

                optimizer.step()

        estimated_MI = backbone.get_mutual_information(
            estimate_dataloader,
            loss,
            self.device,
            marginalize=self.marginalize
        )

        return estimated_MI