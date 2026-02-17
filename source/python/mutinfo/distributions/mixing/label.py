import numpy
import math

from collections import defaultdict, Counter
from collections.abc import Sequence
from scipy.stats import randint
from scipy.stats._distn_infrastructure import rv_frozen, rv_discrete_frozen
from scipy.stats._multivariate import multi_rv_frozen
from typing import Any

from ..tools import BaseMutualInformationTest


class subsampler(multi_rv_frozen):
    """
    Frozen distribution with label data.
    """
    
    def __init__(
        self,
        data: Sequence,
        subset_indices: numpy.ndarray,
        replace: bool=False
    ) -> None:
        self.data = data
        self.subset_indices = subset_indices
        self.replace = replace
        

    def rvs(self, size: int=1) -> Sequence:
        """
        Random variate.

        Parameters
        ----------
        size : int, optional
            Number of samples.

        Returns
        -------
        x : numpy.ndarray
            Random non-repetitive sampling.
        """
            
        length = len(self.subset_indices)
        if not self.replace:
            length = (int(size / length) + 1) * length
        
        indices = numpy.random.choice(length, size=size, replace=self.replace)
        indices = numpy.remainder(indices, len(self.subset_indices)) # TODO: come up with something more effective.
        
        return self.data[self.subset_indices[indices]]


def labeled_dataset_to_subsamplers(
    data: numpy.ndarray,
    labels: numpy.ndarray,
    split_by_labels: bool=True,
) -> dict[Any, subsampler]:
    """
    Convert labeled data into a dict of per-class subsamplers.

    Parameters
    ----------
    data : array_like
        Labeled data.
    labels : array_like

    Returns
    -------
    x : numpy.ndarray
        Random non-repetitive sampling.
    """

    if split_by_labels:
        # Shitty as hell implementation.
        subsamplers = {}
        for label in numpy.unique(labels):
            subsamplers[label] = subsampler(data, numpy.nonzero(labels == label)[0])
    
        return subsamplers
    else:
        return subsampler(data, numpy.arange(0, len(data)))

# TODO: does it belong here?
def torchvision_default_transform(
    dataset,
    dataloader_kwargs: dict={
        "batch_size": 512,
        "shuffle": False,
    },
    to_CHW: bool=False
) -> numpy.ndarray:
    import torch

    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    
    # Extract embeddings
    data = []
    targets = []
    with torch.no_grad():
        for x, y in dataloader:
            data.append(x.cpu().numpy())
            targets.append(y.cpu().numpy())
    
    return numpy.concatenate(data, axis=0), numpy.concatenate(targets, axis=0)

# TODO: move?
def embedding_with_resnet18_backbone(
    dataset,
    checkpoint_path: str=None,
    dataloader_kwargs: dict={
        "batch_size": 512,
        "shuffle": False,
    },
    device: str="cpu",
) -> numpy.ndarray:
    """
    Extract embeddings from CIFAR-10 images using a trained ResNet18 backbone.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input images with shape (N, H, W, C) or (N, C, H, W).
        Values should be in range [0, 1] or [0, 255].
    checkpoint_path : str, optional
        Path to the checkpoint file. If None, uses the best model from resnet18_checkpoints/.
    
    Returns
    -------
    embeddings : numpy.ndarray
        Feature embeddings with shape (N, 512) for ResNet18.
    """
    
    import torch
    import torch.nn as nn
    import torchvision
    from pathlib import Path
    
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Adapt backbone for CIFAR-10 (from supervised_learning_cifar10.py).
    def adapt_backbone_to_CIFAR(backbone):
        backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=(1, 1))
        backbone.maxpool = torch.nn.Identity()
        return backbone
    
    # Create model.
    backbone = torchvision.models.resnet18(num_classes=128)
    backbone = adapt_backbone_to_CIFAR(backbone)
    backbone.fc = nn.Linear(512, 10)  # Match the saved model structure.
    
    # Load weights - strip "backbone." prefix from checkpoint keys.
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
    backbone.load_state_dict(state_dict)
    
    # Remove final classification layer to extract embeddings.
    backbone.fc = nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()

    # Use dataloader for batched processing.
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    
    # Extract embeddings
    embeddings = []
    targets = []
    with torch.no_grad():
        for x, y in dataloader:
            embeddings.append(backbone(x.to(device)).cpu().numpy())
            targets.append(y.cpu().numpy())
    
    return numpy.concatenate(embeddings, axis=0), numpy.concatenate(targets, axis=0) #numpy.asarray(dataset.targets)

# TODO: does it belong here?
def torchvision_labeled_dataset_to_subsamplers(
    dataset,
    transform=torchvision_default_transform,
    split_by_labels: bool=True,
) -> dict[Any, subsampler]:
    """
    Convert torchvision labeled dataset into a dict of per-class subsamplers.

    Parameters
    ----------
    data : array_like
        Labeled data.
    labels : array_like

    Returns
    -------
    x : numpy.ndarray
        Random non-repetitive sampling.
    """

    data, labels = transform(dataset)
    
    return labeled_dataset_to_subsamplers(data, labels, split_by_labels)


class mixed_by_label(multi_rv_frozen, BaseMutualInformationTest):
    def __init__(
        self,
        marginal_distributions: dict[Any, list[multi_rv_frozen | rv_frozen]],
        labels_distribution: rv_discrete_frozen
    ) -> None:
        self._marginal_distributions = marginal_distributions
        self._labels_distribution = labels_distribution

    def rvs(self, size: int=1) -> list:
        """
        Random variate.

        Parameters
        ----------
        size : int, optional
            Number of samples.

        Returns
        -------
        x_1, ..., x_k : numpy.ndarray
            Random sampling.
        """

        labels_tuple = self._labels_distribution.rvs(size=size)
        
        # Cannot come up with a better way to do this.
        sampling = []
        for labels, marginal_distribution in zip(labels_tuple, self._marginal_distributions):
            labels_counts = Counter(labels)
            labels_invargsort = numpy.empty_like(labels)
            labels_invargsort[numpy.argsort(labels, kind="stable")] = numpy.arange(len(labels))

            sampling.append(
                numpy.concatenate(
                    [
                        marginal_distribution[label].rvs(size=labels_counts[label])
                        for label in sorted(labels_counts.keys())
                    ],
                    axis=0
                )[labels_invargsort]
            )

        return tuple(sampling)

    @property
    def mutual_information(self) -> float:
        """
        Mutual information.

        Returns
        -------
        mutual_information : float
            Mutual information.
        """

        if len(self._marginal_distributions) != 2:
            raise ValueError("Mutual information is only defined for pairs of random variables.")
        
        return self._labels_distribution.mutual_information