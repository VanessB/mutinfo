"""PyTorch Lightning module for Flow-based generative models."""

import torch
import functools
import pytorch_lightning as pl
import diffusers
from hydra.utils import instantiate
from mutinfo.distributions.generative.ema import EMA
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class FlowLightningModule(pl.LightningModule):
    """Lightning module for training flow models."""
    
    def __init__(
        self,
        model_config,
        optimizer_config,
        scheduler_config=None,
        prior_dim=2,
        use_ema=False,
        ema_decay=0.9999,
        log_samples=True,
        log_every_n_epochs=10,
        n_samples_to_log=16,
        image_shape=None,
    ):
        """
        Initialize the Flow Lightning Module.
        
        Args:
            model_config: Hydra config for instantiating the flow model
            optimizer_config: Hydra config for instantiating the optimizer
            scheduler_config: Optional Hydra config for learning rate scheduler
            prior_dim: Dimensionality of the prior distribution (standard normal)
            use_ema: Whether to use Exponential Moving Average for model weights
            ema_decay: Decay rate for EMA (default: 0.9999)
            log_samples: Whether to log samples during training/validation
            log_every_n_epochs: Log samples every N epochs
            n_samples_to_log: Number of samples to log
            image_shape: Shape for reshaping samples (e.g., (1, 28, 28) for MNIST, (3, 32, 32) for CIFAR)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Instantiate the flow model
        self.model = instantiate(model_config)
        
        # Setup EMA if requested
        self.use_ema = use_ema
        if self.use_ema:
            self.ema = EMA(self.model, decay=ema_decay)
        else:
            self.ema = None
        
        # Store configs for optimizer/scheduler (instantiated in configure_optimizers)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        
        # Prior distribution (standard normal)
        self.prior_dim = prior_dim
        
        # Sample logging configuration
        self.log_samples = log_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.n_samples_to_log = n_samples_to_log
        self.image_shape = image_shape
        
        # Determine data type from prior_dim and image_shape
        if image_shape is not None:
            if len(image_shape) == 3:
                if image_shape[0] == 1:
                    self.data_type = 'mnist'  # Grayscale image
                elif image_shape[0] == 3:
                    self.data_type = 'cifar'  # RGB image
                else:
                    self.data_type = 'image'
            else:
                self.data_type = 'image'
        elif prior_dim == 2:
            self.data_type = '2d'
        elif prior_dim == 784:
            self.data_type = 'mnist'
            self.image_shape = (1, 28, 28)
        elif prior_dim == 3072:
            self.data_type = 'cifar'
            self.image_shape = (3, 32, 32)
        else:
            self.data_type = 'other'
        
        # Loss function
        self.loss_fn = torch.nn.MSELoss()
    
    def forward(self, t, x_t):
        """Forward pass through the flow model."""
        if isinstance(self.model, diffusers.UNet2DModel):
            # For diffusers UNet2DModel, reshape input to (B, C, H, W)
            t, x_t = self._process_t_and_x_for_diffusers_unet(t, x_t)
            B = x_t.shape[0]
            out = self.model(x_t, t).sample
            return out.view(B, -1)  # Flatten back to (B, prior_dim)
        return self.model(t=t, x_t=x_t)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # batch is (x_0, x_1) where x_0 is data, x_1 is prior sample
        x_0, x_1 = batch
        
        # Sample random time t ~ Uniform[0, 1]
        t = torch.rand(len(x_0), 1, device=self.device)
        
        # Compute interpolation x_t = (1-t)*x_0 + t*x_1
        x_t = (1 - t) * x_0 + t * x_1
        
        # Target is the velocity field dx_t = x_1 - x_0
        dx_t = x_1 - x_0
        
        # Predict velocity field
        pred_dx_t = self(t, x_t)
        
        # Compute loss
        loss = self.loss_fn(pred_dx_t, dx_t)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_before_backward(self, loss):
        if self.use_ema:
            self.ema.update(self.model)
    
    def on_train_epoch_end(self):
        """Log denoised samples at the end of training epoch."""
        if not self.log_samples:
            return
        
        if self.current_epoch % self.log_every_n_epochs != 0:
            return
        
        # Generate denoised samples
        try:
            with torch.no_grad():
                # Sample some data points and add noise
                n_samples = min(self.n_samples_to_log, 16)
                
                # Create noisy samples at t=0.5
                x_prior = self.sample_prior(n_samples)
                t = torch.ones(n_samples, 1, device=self.device) * 0.5
                
                # Denoise using the model
                model = self.ema.module if self.use_ema else self.model
                if isinstance(model, diffusers.UNet2DModel):
                    t_processed, x_prior_processed = self._process_t_and_x_for_diffusers_unet(t, x_prior)
                    B = x_prior_processed.shape[0]
                    velocity = model(x_prior_processed, t_processed)
                    velocity = velocity.sample.view(B, -1)
                else:
                    velocity = model(t, x_prior)
                x_denoised = x_prior - velocity * 0.5
                
                # Log based on data type
                fig = self._create_sample_figure(x_denoised, title=f'Denoised Samples (Epoch {self.current_epoch})')
                
                if fig is not None:
                    # Log to wandb
                    import wandb
                    self.logger.log_image(
                        key='train/denoised_samples',
                        images=[wandb.Image(fig)]
                    )
                    plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not log training samples: {e}")
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x_0, x_1 = batch
        
        t = torch.rand(len(x_0), 1, device=self.device)
        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0
        
        pred_dx_t = self(t, x_t)
        loss = self.loss_fn(pred_dx_t, dx_t)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Also compute validation loss with EMA model if available
        if self.use_ema:
            with torch.no_grad():
                if isinstance(self.ema.module, diffusers.UNet2DModel):
                    t_processed, x_t_processed = self._process_t_and_x_for_diffusers_unet(t, x_t)
                    B = x_t_processed.shape[0]
                    ema_pred_dx_t = self.ema.module(x_t_processed, t_processed)
                    ema_pred_dx_t = ema_pred_dx_t.sample.view(B, -1)
                else:
                    ema_pred_dx_t = self.ema.module(t, x_t)
                ema_loss = self.loss_fn(ema_pred_dx_t, dx_t)
                self.log('val_loss_ema', ema_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Log generated samples at the end of validation epoch."""
        if not self.log_samples:
            return
        
        if self.current_epoch % self.log_every_n_epochs != 0:
            return
        
        # Generate samples from prior
        try:
            with torch.no_grad():
                n_samples = min(self.n_samples_to_log, 64)
                samples = self.generate_samples(
                    n_samples=n_samples,
                    num_steps=50,  # Use fewer steps for faster logging
                    use_ema=self.use_ema,
                )
                
                # Log based on data type
                fig = self._create_sample_figure(samples, title=f'Generated Samples (Epoch {self.current_epoch})')
                
                if fig is not None:
                    # Log to wandb
                    import wandb
                    self.logger.log_image(
                        key='val/generated_samples',
                        images=[wandb.Image(fig)]
                    )
                    plt.close(fig)
                
                # For 2D data, also log the ODE trajectory
                if self.data_type == '2d':
                    fig_trajectory = self._log_ode_trajectory(n_samples=min(1000, n_samples * 10))
                    if fig_trajectory is not None:
                        import wandb
                        self.logger.log_image(
                            key='val/ode_trajectory',
                            images=[wandb.Image(fig_trajectory)]
                        )
                        plt.close(fig_trajectory)
        except Exception as e:
            print(f"Warning: Could not log validation samples: {e}")
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Instantiate optimizer
        optimizer = instantiate(self.optimizer_config)(params=self.parameters())
        assert not isinstance(optimizer, functools.partial)
        
        if self.scheduler_config is not None:
            # Instantiate scheduler
            scheduler = instantiate(self.scheduler_config)(optimizer=optimizer)
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        return optimizer
    
    def sample_prior(self, n_samples):
        """Sample from the prior distribution (standard normal)."""
        return torch.randn(n_samples, self.prior_dim, device=self.device)
    
    def _process_t_and_x_for_diffusers_unet(self, t, x_t):
        """Process time and input for diffusers UNet2DModel."""
        B = x_t.shape[0]
        if self.image_shape is not None:
            C, H, W = self.image_shape
        else:
            # Try to infer square image shape
            dim = int(np.sqrt(x_t.shape[1]))
            C, H, W = 1, dim, dim  # Assume grayscale if unknown
        
        x_t_reshaped = x_t.view(B, C, H, W)
        t_scaled = (t * 1000).squeeze(1)  # Scale time to [0, 1000] as expected by diffusers
        
        return t_scaled, x_t_reshaped
    
    def _create_sample_figure(self, samples, title='Samples'):
        """
        Create a matplotlib figure for visualizing samples.
        
        Args:
            samples: Tensor of samples (n_samples, dim)
            title: Title for the figure
            
        Returns:
            matplotlib figure or None
        """
        samples_np = samples.cpu().numpy()
        
        if self.data_type == '2d':
            # 2D scatter plot for make_moons, etc.
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, s=10)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            return fig
        
        elif self.data_type in ['mnist', 'cifar', 'image']:
            # Image grid for MNIST, CIFAR, etc.
            n_samples = len(samples_np)
            
            # Reshape samples
            if self.image_shape is not None:
                samples_img = samples_np.reshape(-1, *self.image_shape)
            else:
                # Try to infer square image shape
                dim = int(np.sqrt(samples_np.shape[1]))
                samples_img = samples_np.reshape(-1, 1, dim, dim)
            
            # Normalize to [0, 1] for visualization
            samples_img = (samples_img - samples_img.min()) / (samples_img.max() - samples_img.min() + 1e-8)
            
            # Create grid
            n_cols = min(8, n_samples)
            n_rows = (n_samples + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
            if n_samples == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for i in range(len(axes)):
                ax = axes[i]
                if i < n_samples:
                    # Transpose from (C, H, W) to (H, W, C) for matplotlib
                    img = samples_img[i].transpose(1, 2, 0)
                    
                    # Handle grayscale vs RGB
                    if img.shape[2] == 1:
                        ax.imshow(img.squeeze(), cmap='gray')
                    else:
                        ax.imshow(img)
                    ax.axis('off')
                else:
                    ax.axis('off')
            
            plt.suptitle(title)
            plt.tight_layout()
            return fig
        
        else:
            # For other data types, just return None
            return None
    
    def _log_ode_trajectory(self, n_samples=1000, n_steps=8):
        """
        Create a figure showing the ODE trajectory from prior to data.
        Shows intermediate steps of the generative process.
        
        Args:
            n_samples: Number of samples to visualize
            n_steps: Number of intermediate steps to show
            
        Returns:
            matplotlib figure or None
        """
        if self.data_type != '2d':
            return None
        
        try:
            with torch.no_grad():
                # Choose model
                model = self.ema.module if (self.use_ema and self.ema is not None) else self.model
                
                # Start from prior
                x = self.sample_prior(n_samples)
                
                # Store trajectory
                trajectory = [x.cpu().numpy()]
                time_steps = torch.linspace(1.0, 0.0, n_steps + 1)
                
                # Integrate ODE
                dt = 1.0 / n_steps
                for step in range(n_steps):
                    t = torch.ones(n_samples, 1, device=self.device) * (1 - step * dt)
                    
                    # Check if model has step method (like FlowMLP)
                    if hasattr(model, 'step'):
                        t_start = time_steps[step]
                        t_end = time_steps[step + 1]
                        x = model.step(x, t_start, t_end)
                    else:
                        dx = model(t, x)
                        x = x - dx * dt
                    
                    trajectory.append(x.cpu().numpy())
                
                # Create figure with subplots
                fig, axes = plt.subplots(1, n_steps + 1, figsize=(3 * (n_steps + 1), 3), 
                                        sharex=True, sharey=True)
                
                # Plot each timestep
                for i, (ax, t_val) in enumerate(zip(axes, time_steps)):
                    samples = trajectory[i]
                    ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
                    ax.set_title(f't = {t_val:.2f}')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(-3.0, 3.0)
                    ax.set_ylim(-3.0, 3.0)
                    ax.set_aspect('equal')
                    
                    if i == 0:
                        ax.set_ylabel('y')
                    ax.set_xlabel('x')
                
                plt.suptitle(f'ODE Trajectory (Prior → Data, Epoch {self.current_epoch})', fontsize=14)
                plt.tight_layout()
                
                return fig
        except Exception as e:
            print(f"Warning: Could not create ODE trajectory plot: {e}")
            return None
    
    def generate_samples(self, n_samples=None, num_steps=100, use_ema=True, prior_samples=None):
        """
        Generate samples by solving the ODE from prior to data.
        
        Args:
            n_samples: Number of samples to generate
            num_steps: Number of integration steps
            use_ema: Whether to use EMA model for generation (if available)
            prior_samples: Optional prior samples to start from (if not using model's prior)
            
        Returns:
            Generated samples
        """
        self.eval()

        assert n_samples is not None or prior_samples is not None, "Must provide n_samples or prior_samples"
        
        # Choose which model to use for generation
        if use_ema and self.use_ema:
            model = self.ema.module
        else:
            model = self.model
        
        with torch.no_grad():
            # Start from prior samples
            if prior_samples is not None:
                x = prior_samples.to(self.device)
                n_samples = x.shape[0]
            else:
                x = self.sample_prior(n_samples)
            
            # Integrate from t=1 (prior) to t=0 (data) using midpoint method
            time_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
            for step in range(num_steps):
                t_start = time_steps[step]
                t_end = time_steps[step + 1]
                t_start_batch = t_start.view(1, 1).expand(x.shape[0], 1)
                try:
                    x = x + (t_end - t_start) * self._forward_model(model, t=t_start_batch + (t_end - t_start) / 2, 
                                                                    x_t=x + self._forward_model(model, x_t=x, t=t_start_batch) * (t_end - t_start) / 2)
                except:
                    raise ValueError(f"Incompatible shape: x={x.shape}, t_start_batch={t_start_batch.shape}, model expected input shape: {model.forward.__annotations__}")
            return x
    
    def _forward_model(self, model, t, x_t):
        """Helper to call model forward, handling diffusers UNet if needed."""
        if isinstance(model, diffusers.UNet2DModel):
            t_processed, x_processed = self._process_t_and_x_for_diffusers_unet(t, x_t)
            out = model(x_processed, t_processed).sample.view(x_t.shape[0], -1)
            return out
        return model(t=t, x_t=x_t)
