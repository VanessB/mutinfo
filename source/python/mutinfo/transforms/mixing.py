import torch

def permute_pairs_preserve_labels(
    samples: torch.Tensor,
    labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    unique_labels = labels.unique()

    subsets_indices = []
    permuted_subsets_indices = []
    for label in unique_labels:
        subset_indices = (labels == label).nonzero()
        subsets_indices.append(subset_indices)
        permuted_subsets_indices.append(subset_indices[torch.randperm(subset_indices.shape[0])])

    indices = torch.cat(subsets_indices, axis=0)
    permuted_indices = torch.cat(permuted_subsets_indices, axis=0)

    return samples[indices], samples[permuted_indices]