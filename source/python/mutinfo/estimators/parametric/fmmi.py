import math
import numpy
import torch
import fmmi

from collections.abc import Callable

from sklearn.model_selection import train_test_split

from ..base import MutualInformationEstimator

from fmmi.utils.modules import VelocityModelMLP


def conditional_VelocityModelMLP_wrapper(
    x_shape: tuple[int],
    y_shape: tuple[int],
    backbone_factory: torch.nn.Module=VelocityModelMLP,
    **kwargs,
) -> torch.nn.Module:
    condition_dim = math.prod(x_shape[1:])
    input_dim = math.prod(y_shape[1:])

    return backbone_factory(input_dim=input_dim, condition_dim=condition_dim, **kwargs)

def joint_VelocityModelMLP_wrapper(
    x_shape: tuple[int],
    y_shape: tuple[int],
    backbone_factory: torch.nn.Module=VelocityModelMLP,
    **kwargs,
) -> torch.nn.Module:
    condition_dim = 0
    input_dim = math.prod(y_shape[1:]) + math.prod(x_shape[1:])

    return backbone_factory(input_dim=input_dim, condition_dim=condition_dim, **kwargs)


class FMMI(MutualInformationEstimator):
    def __init__(
        self,
        estimator_factory: Callable[[], fmmi.estimator.mi.FMMI]=None,
        backbone_factory: Callable[[], torch.nn.Module]=None,
        optimizer_factory: Callable[[], torch.optim.Optimizer]=None,
        n_train_steps: int=10000,
        train_batch_size: int=512,
        estimate_batch_size: int=512,
        estimate_size: float | int=0.5,
        exact_divergence: bool=False,
        swap_x_y: bool=False,
        device: str="cpu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.estimator_factory = estimator_factory
        if self.estimator_factory is None:
            self.estimator_factory = fmmi.estimator.mi.cFMMI

        self.backbone_factory = backbone_factory
        if self.backbone_factory is None:
            self.backbone_factory = conditional_VelocityModelMLP_wrapper

        self.optimizer_factory = optimizer_factory
        if self.optimizer_factory is None:
            self.optimizer_factory = lambda parameters : torch.optim.AdamW(parameters, lr=1.0e-3, weight_decay=1.0e-5)

        self.n_train_steps = n_train_steps
        self.train_batch_size = train_batch_size
        self.estimate_batch_size = estimate_batch_size
        self.estimate_size = estimate_size
        self.exact_divergence = exact_divergence
        self.swap_x_y = swap_x_y
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

        if self.swap_x_y:
            x, y = y, x

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

        estimator = self.estimator_factory(
            backbone = self.backbone_factory(
                x.shape,
                y.shape,
            ),
        ).to(self.device)
        optimizer = self.optimizer_factory(estimator.parameters())

        step = 0
        while step < self.n_train_steps:
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                t = torch.rand(x.shape[0], device=self.device)
                
                estimator.get_batch_loss(x, y, t).backward()

                optimizer.step()
                step += 1

        estimated_MI = estimator.estimate(estimate_dataloader, device=self.device, exact_divergence=self.exact_divergence)

        return max(estimated_MI, 0.0)