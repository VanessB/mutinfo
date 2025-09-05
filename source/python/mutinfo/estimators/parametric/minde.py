import pytorch_lightning as pl
from omegaconf import DictConfig
from . import infosedd_utils
from . import minde_utils
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from . import graph_lib
from . import noise_lib
from . import sde_lib
import time
import psutil
import hydra
import torch
import os
import numpy


class GenerativeMIEstimator(pl.LightningModule):
    def __init__(self, 
                 name: str,
                 backbone_factory: callable,
                 optimizer_factory: callable,
                 trainer: pl.Trainer,
                 logger = None,
                 sde : sde_lib.VP_SDE = None,
                 noise: noise_lib.Noise = None,
                 graph: graph_lib.Graph = None,
                 train_batch_size: int=512,
                 estimate_batch_size: int=512,
                 estimate_fraction: float=0.5,
                 variant: str="c",
                 is_parametric_marginal: bool=False,
                 sampling_eps: float=0.001):
        super().__init__()
        self.estimator_name = name.lower()
        self.sde = sde
        self.noise = noise
        self.graph = graph
        self.variant = variant
        self.backbone_factory = backbone_factory
        self.optimizer_factory = optimizer_factory
        self.trainer = trainer
        self.train_batch_size = train_batch_size
        self.estimate_batch_size = estimate_batch_size
        self.estimate_fraction = estimate_fraction
        self.sampling_eps = sampling_eps
        self.is_parametric_marginal = is_parametric_marginal

    def loss(self, x, y):
        if 'infosedd' in self.estimator_name:
            loss = self._loss(self.backbone, x, y)
        elif 'minde' in self.estimator_name:
            backbone_score_forward = partial(minde_utils.score_forward, self.backbone)
            loss = self.sde.train_step(x, y, backbone_score_forward).mean()
        else:
            raise NotImplementedError(f"Estimator {self.estimator_name} not implemented.")
        
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        loss = self.loss(x, y)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_start(self):
        self.epoch_start_time = time.time()
        self.mi_values = []

    def on_validation_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        gpu_max_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        self.results["runtime"].append(epoch_time)
        self.results["memory"].append({"cpu_mb": memory_mb, "gpu_mb": gpu_memory_mb, "gpu_max_mb": gpu_max_memory_mb})
        if self.mi_values and len(self.mi_values) > 0:
            self.results["mi_history"].append({'epoch': self.current_epoch,\
                                                'mean_mi': torch.mean(torch.tensor(self.mi_values)).item(),
                                                'std_mi': torch.std(torch.tensor(self.mi_values)).item()})

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        if 'infosedd' in self.estimator_name:
            mi = self._mutinfo_fn(self.backbone, x, y)
        elif 'minde' in self.estimator_name:
            mi = self._mutinfo_fn(self.backbone, x, y)
        else:
            raise NotImplementedError(f"Estimator {self.estimator_name} not implemented.")
        self.log("MI", mi, on_step=False, on_epoch=True, prog_bar=True)
        self.mi_values.append(mi)

        return mi

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.backbone.parameters())
        return optimizer
    
    def _setup_estimator(self):
        self.results = {"mi_history": [], "runtime": [], "memory": []}
        self.epoch_start_time = None
        if 'infosedd' in self.estimator_name:
            self.graph.dim = self.alphabet_size
            self.alphabet_size += 1
            print(self.graph.dim, self.alphabet_size)
            self._loss = infosedd_utils.get_loss_fn(noise=self.noise,\
                                                    graph=self.graph,\
                                                    train=True,\
                                                    is_parametric_marginal=self.is_parametric_marginal,\
                                                    variant=self.variant,\
                                                    sampling_eps=self.sampling_eps)
            self._mutinfo_fn = infosedd_utils.get_mutinfo_step_fn(self.graph, self.noise, self.variant)
            self.is_parametric_marginal = self.is_parametric_marginal
            self.backbone = self.backbone_factory(X_shape=self.x_shape, Y_shape=self.y_shape, alphabet_size=self.alphabet_size).to(self.device)
        elif 'minde' in self.estimator_name:
            self._mutinfo_fn = minde_utils.get_mutinfo_step_fn(self.sde, self.sde.importance_sampling, self.variant, self.sampling_eps)
            self.backbone = self.backbone_factory(X_shape=self.x_shape, Y_shape=self.y_shape).to(self.device)
        else:
            raise NotImplementedError(f"Estimator {self.estimator_name} not implemented.")
    
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

        if "infosedd" in self.estimator_name:
            x = x*255
            y = y*255

        self.x_shape = x.shape[1:]
        self.y_shape = y.shape[1:]

        self.total_X_size = numpy.prod(x.shape[1:])
        self.total_Y_size = numpy.prod(y.shape[1:])

        if self.estimate_fraction is None:
            train_x, estimate_x, train_y, estimate_y = x, x, y, y
        else:
            train_x, estimate_x, train_y, estimate_y = train_test_split(x, y, test_size=self.estimate_fraction)

        if "infosedd" in self.estimator_name:
            train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(train_x, dtype=torch.long),
                torch.tensor(train_y, dtype=torch.long),
            )
        else:
            train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(train_x, dtype=torch.float32),
                torch.tensor(train_y, dtype=torch.float32),
            )

        if "minde" in self.estimator_name:
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            train_x = scaler_x.fit_transform(train_x.reshape(-1, self.total_X_size))
            train_y = scaler_y.fit_transform(train_y.reshape(-1, self.total_Y_size))
            estimate_x = scaler_x.transform(estimate_x.reshape(-1, self.total_X_size))
            estimate_y = scaler_y.transform(estimate_y.reshape(-1, self.total_Y_size))            
            
        if "infosedd" in self.estimator_name:
            estimate_dataset = torch.utils.data.TensorDataset(
                torch.tensor(estimate_x, dtype=torch.long),
                torch.tensor(estimate_y, dtype=torch.long),
            )
        else:
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

        if "infosedd" in self.estimator_name:
            self.alphabet_size = int(max(x.max(), y.max()) + 1)
            if isinstance(self.graph, graph_lib.Absorbing):
                self.alphabet_size += 1

        self._setup_estimator()

        

        print(f"Starting run with a model with {sum(p.numel() for p in self.backbone.parameters())} parameters")

        self.trainer.fit(self, 
                    train_dataloaders=train_dataloader,
                    val_dataloaders=estimate_dataloader,
                    )
                    

        return max(torch.mean(torch.tensor(self.mi_values)).item(), 0.0) 