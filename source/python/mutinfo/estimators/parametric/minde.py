import pytorch_lightning as pl
from omegaconf import DictConfig
from . import minde_utils
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from . import sde_lib
from . import ema
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
                 train_batch_size: int=512,
                 estimate_batch_size: int=512,
                 estimate_fraction: float=0.5,
                 variant: str="c",
                 is_parametric_marginal: bool=False,
                 sampling_eps: float=0.001,
                 use_ema: bool=True,
                 ema_decay: float=0.9999,
                 ckpt_path: str = None,):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.estimator_name = name.lower()
        self.sde = sde
        self.variant = variant
        self.backbone_factory = backbone_factory
        self.optimizer_factory = optimizer_factory
        self.trainer = trainer
        self.train_batch_size = train_batch_size
        self.estimate_batch_size = estimate_batch_size
        self.estimate_fraction = estimate_fraction
        self.sampling_eps = sampling_eps
        self.is_parametric_marginal = is_parametric_marginal
        self.use_ema = use_ema
        self.ema_decay = ema_decay
    
    def log_mnist_image(self, img, name):
        return
        import wandb
        img = self.inverse_scaler(img)
        img = img.reshape(img.shape[0], 2, int(numpy.sqrt(img.shape[1]//2)), int(numpy.sqrt(img.shape[1]//2)))
        # denormalize
        img = img.clip(0,1)
        img = img * 255
        # Assuming x_denoised has shape (batch_size, 2, h, w)
        x_img = img[0, 0].detach().cpu().numpy()  # First image from first batch
        y_img = img[0, 1].detach().cpu().numpy()  # Second image from first batch

        try:
            self.logger.experiment.log({
                f"{name}_x": wandb.Image(x_img),
                f"{name}_y": wandb.Image(y_img),
                "global_step": self.global_step
            })
        except:
            raise ValueError(f"Could not log images to wandb, x_img shape is {x_img.shape}, y_img shape is {y_img.shape}, x_denoised shape is {x_img.shape}")

    def loss(self, x, y):
        
        if 'minde' in self.estimator_name:
            backbone_score_forward = partial(minde_utils.score_forward, self.backbone)
            loss, ret_dict = self.sde.train_step(x, y, backbone_score_forward, debug=False)
            loss = loss.mean()
            x_denoised = ret_dict['x_denoised']
            xy_t = ret_dict['xy_t']
            noise = ret_dict['noise']
            score = torch.abs(ret_dict['score'])
            
            # Log both images separately with wandb
            if hasattr(self, 'logger') and self.logger is not None and hasattr(self.logger, 'experiment'):
                if self.global_step % 1000 == 0:
                    self.log_mnist_image(x_denoised, "x_denoised")
                    self.log_mnist_image(xy_t, "xy_t")
                    self.log_mnist_image(noise, "noise")
                    self.log_mnist_image(score, "score")
                    self.log("score_norm", ret_dict['score_norm'], on_step=True, on_epoch=False, prog_bar=True)
                    self.log("noise_norm", ret_dict['noise_norm'], on_step=True, on_epoch=False, prog_bar=True)

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
    
    def _sample(self, batch_size):
        if 'minde' in self.estimator_name:
            return self.sde.sample(batch_size, self.total_X_size, self.total_Y_size, device=self.device)
    
    def on_before_backward(self, loss):
        assert self.use_ema, "EMA not used, cannot update."
        if self.ema_backbone:
            self.ema_backbone.update(self.backbone)
        else:
            raise ValueError("EMA backbone not initialized.")

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
        if 'minde' in self.estimator_name:
            score = self.ema_backbone.module if self.use_ema else self.score
            mi = self._mutinfo_fn(score, x, y)
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
        if 'minde' in self.estimator_name:
            self._mutinfo_fn = minde_utils.get_mutinfo_step_fn(self.sde, self.sde.importance_sampling, self.variant, self.sampling_eps)
            self.backbone = self.backbone_factory(X_shape=self.x_shape, Y_shape=self.y_shape).to(self.device)
            if self.use_ema:
                self.ema_backbone = ema.EMA(self.backbone, decay=self.ema_decay)
            else:
                self.ema_backbone = None
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

        self.x_shape = x.shape[1:]
        self.y_shape = y.shape[1:]

        self.total_X_size = numpy.prod(x.shape[1:])
        self.total_Y_size = numpy.prod(y.shape[1:])

        if self.estimate_fraction is None:
            train_x, estimate_x, train_y, estimate_y = x, x, y, y
        else:
            train_x, estimate_x, train_y, estimate_y = train_test_split(x, y, test_size=self.estimate_fraction)

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_x, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
        )
            

        if "minde" in self.estimator_name:
            self.scaler = lambda x: x
            self.inverse_scaler = lambda x: x
            train_x = self.scaler(train_x)
            train_y = self.scaler(train_y)
            estimate_x = self.scaler(estimate_x)
            estimate_y = self.scaler(estimate_y)
            train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(train_x, dtype=torch.float32),
                torch.tensor(train_y, dtype=torch.float32),
            )
            estimate_dataset = torch.utils.data.TensorDataset(
                torch.tensor(estimate_x, dtype=torch.float32),
                torch.tensor(estimate_y, dtype=torch.float32),
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

        self._setup_estimator()

        print(f"Starting run with a model with {sum(p.numel() for p in self.backbone.parameters())} parameters")

        self.trainer.fit(self, 
                    train_dataloaders=train_dataloader,
                    val_dataloaders=estimate_dataloader,
                    ckpt_path=self.ckpt_path
                    )
                    

        return max(torch.mean(torch.tensor(self.mi_values)).item(), 0.0) 