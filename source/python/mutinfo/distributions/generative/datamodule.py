"""PyTorch Lightning DataModule for MNIST dataset."""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


class MNISTFlowDataModule(pl.LightningDataModule):
    """DataModule for MNIST dataset for flow-based generative modeling."""
    
    def __init__(
        self,
        data_dir: str = './data',
        batch_size: int = 256,
        num_workers: int = 4,
        flatten: bool = True,
        normalize: bool = True,
        prior_type: str = 'normal',  # 'normal' or 'uniform'
    ):
        """
        Initialize MNIST DataModule for flow training.
        
        Args:
            data_dir: Directory to store/load MNIST data
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            flatten: Whether to flatten images to vectors
            normalize: Whether to normalize to [0, 1] or [-1, 1]
            prior_type: Type of prior distribution ('normal' or 'uniform')
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.flatten = flatten
        self.normalize = normalize
        self.prior_type = prior_type
        
        # Setup transforms
        transform_list = [transforms.ToTensor()]
        if normalize:
            # Normalize to [-1, 1]
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        
        self.transform = transforms.Compose(transform_list)
        
    def prepare_data(self):
        """Download MNIST data."""
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        """Setup train/val/test datasets."""
        if stage == 'fit' or stage is None:
            mnist_train = datasets.MNIST(
                self.data_dir, 
                train=True, 
                transform=self.transform
            )
            
            # Split into train and validation
            train_size = int(0.9 * len(mnist_train))
            val_size = len(mnist_train) - train_size
            
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(
                mnist_train, [train_size, val_size]
            )
        
        if stage == 'test' or stage is None:
            self.mnist_test = datasets.MNIST(
                self.data_dir,
                train=False,
                transform=self.transform
            )
    
    def _sample_prior(self, size):
        """Sample from prior distribution."""
        if self.prior_type == 'normal':
            return torch.randn(size)
        elif self.prior_type == 'uniform':
            return torch.rand(size) * 2 - 1  # Uniform in [-1, 1]
        else:
            raise ValueError(f"Unknown prior_type: {self.prior_type}")
    
    def _collate_fn(self, batch):
        """
        Custom collate function that pairs data samples with prior samples.
        
        Returns:
            (x_0, x_1) where x_0 is data and x_1 is prior sample
        """
        # Extract images and labels
        images, labels = zip(*batch)
        images = torch.stack(images)
        
        # Flatten if needed
        if self.flatten:
            batch_size = images.shape[0]
            images = images.view(batch_size, -1)
        
        # Sample from prior (same shape as data)
        prior_samples = self._sample_prior(images.shape)
        
        return images, prior_samples
    
    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )


class CustomFlowDataModule(pl.LightningDataModule):
    """DataModule for custom datasets (e.g., make_moons, 2D distributions)."""
    
    def __init__(
        self,
        data_generator,
        prior_generator,
        n_train_samples: int = 50000,
        n_val_samples: int = 10000,
        n_test_samples: int = 10000,
        batch_size: int = 256,
        num_workers: int = 0,
    ):
        """
        Initialize custom DataModule.
        
        Args:
            data_generator: Callable that generates data samples, e.g., make_moons
            prior_generator: Callable that generates prior samples
            n_train_samples: Number of training samples
            n_val_samples: Number of validation samples
            n_test_samples: Number of test samples
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
        """
        super().__init__()
        self.save_hyperparameters(ignore=['data_generator', 'prior_generator'])
        self.data_generator = data_generator
        self.prior_generator = prior_generator
        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples
        self.n_test_samples = n_test_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        """Generate datasets."""
        if stage == 'fit' or stage is None:
            # Generate training data
            train_data = self.data_generator(self.n_train_samples)
            # Handle tuple returns (e.g., from sklearn make_moons)
            if isinstance(train_data, tuple):
                train_data = train_data[0]
            
            train_prior = self.prior_generator(self.n_train_samples, train_data.shape[1])
            self.train_dataset = TensorDataset(
                torch.FloatTensor(train_data),
                torch.FloatTensor(train_prior)
            )
            
            # Generate validation data
            val_data = self.data_generator(self.n_val_samples)
            if isinstance(val_data, tuple):
                val_data = val_data[0]
                
            val_prior = self.prior_generator(self.n_val_samples, val_data.shape[1])
            self.val_dataset = TensorDataset(
                torch.FloatTensor(val_data),
                torch.FloatTensor(val_prior)
            )
        
        if stage == 'test' or stage is None:
            # Generate test data
            test_data = self.data_generator(self.n_test_samples)
            if isinstance(test_data, tuple):
                test_data = test_data[0]
                
            test_prior = self.prior_generator(self.n_test_samples, test_data.shape[1])
            self.test_dataset = TensorDataset(
                torch.FloatTensor(test_data),
                torch.FloatTensor(test_prior)
            )
    
    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
