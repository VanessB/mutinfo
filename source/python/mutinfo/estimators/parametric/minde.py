import math
import numpy
import torch
import torchkld
import wandb
import numpy as np

from collections.abc import Callable
from copy import deepcopy

from sklearn.model_selection import train_test_split

from ..base import MutualInformationEstimator

_EPS = 1.0e-6

import math
import numpy
import torch

from collections.abc import Callable

from sklearn.model_selection import train_test_split
from .importance_sampling_utils import get_normalizing_constant, sample_importance_sampling

from ..base import MutualInformationEstimator

_EPS = 1.0e-6

class MINDE(MutualInformationEstimator):
    def __init__(
        self,
        backbone_factory: Callable[[], torchkld.mutual_information.MINE]=None,
        optimizer_factory: Callable[[], torch.optim.Optimizer]=None,
        n_train_steps: int=10000,
        train_batch_size: int=512,
        estimate_batch_size: int=512,
        estimate_fraction: float=0.5,
        device: str="cpu",
        use_wandb: bool=False,
        wandb_project: str=None,
        wandb_config: dict=None,
        diffusion_config: dict=None,
        log_every_n_steps: int=1000,
        log_generated_samples: bool=True,
        n_generated_samples: int=4,
        generation_steps: int=100,
        save_generation_process: bool=True,
        variant: str="j",
        sigma: float=1.0,
        use_ema: bool=True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.variant = variant
        self.sigma = sigma
        self.use_ema = use_ema

        self.backbone_factory = backbone_factory
        if self.backbone_factory is None:
            self.backbone_factory = GenericMLPDiffuser

        self.optimizer_factory = optimizer_factory
        if self.optimizer_factory is None:
            self.optimizer_factory = lambda parameters : torch.optim.Adam(parameters, lr=1.0e-3)

        self.n_train_steps = n_train_steps
        self.train_batch_size = train_batch_size
        self.estimate_batch_size = estimate_batch_size
        self.estimate_fraction = estimate_fraction
        self.device = device

        if diffusion_config is None:
            self.diffusion_config = {
                "beta_min": 0.1,
                "beta_max": 20.0,
            }
        else:
            self.diffusion_config = deepcopy(diffusion_config)
        
        # Wandb configuration
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_config = wandb_config or {}
        self.log_every_n_steps = log_every_n_steps
        self.log_generated_samples = log_generated_samples
        self.n_generated_samples = n_generated_samples
        self.generation_steps = generation_steps
        self.save_generation_process = save_generation_process
        
        if self.use_wandb:
            # Initialize wandb with configuration
            config = {
                "n_train_steps": self.n_train_steps,
                "train_batch_size": self.train_batch_size,
                "estimate_batch_size": self.estimate_batch_size,
                "estimate_fraction": self.estimate_fraction,
                "device": self.device,
                **self.wandb_config
            }
            wandb.init(project=self.wandb_project, config=config)

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

        self.x_shape = x.shape[1:]
        self.y_shape = y.shape[1:]

        self.total_X_size = np.prod(x.shape[1:])
        self.total_Y_size = np.prod(y.shape[1:])

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

        backbone = self.backbone_factory(x.shape[1:], y.shape[1:]).to(self.device)
        self.backbone = backbone
        if self.use_ema:
            self.ema = EMA(backbone, decay=0.9999)
        optimizer = self.optimizer_factory(backbone.parameters())

        step = 0
        while step < self.n_train_steps:
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                loss, output_dict = self.loss(
                    x,
                    y
                )
                loss.mean().backward()

                self.log(output_dict, step=step)

                optimizer.step()
                step += 1
                if self.use_ema:
                    self.ema.update(backbone)

        mi, mi_sigma = self.get_mutual_information(
            estimate_dataloader,
            self.device,
        )

        if self.use_wandb:
            wandb.log({"final_mutual_information": max(mi,0.0)})
            wandb.log({"final_mutual_information_sigma": max(mi_sigma,0.0)})
            wandb.finish()

        return max(mi, 0.0) 
    
    def log(self, output_dict, step):
        """Log training metrics to wandb if enabled."""
        if self.use_wandb:
            # Extract scalar values for logging
            log_dict = {
                "step": step,
                "loss": output_dict["loss"].mean().item(),
            }
            
            # Log additional metrics if available
            if "t" in output_dict:
                log_dict["t_mean"] = output_dict["t"].mean().item()
                log_dict["t_std"] = output_dict["t"].std().item()
            
            if "score" in output_dict:
                log_dict["score_mean"] = output_dict["score"].mean().item()
                log_dict["score_std"] = output_dict["score"].std().item()
            
            if "noise" in output_dict:
                log_dict["noise_mean"] = output_dict["noise"].mean().item()
                log_dict["noise_std"] = output_dict["noise"].std().item()
            
            # Log mask coverage (how much data is being diffused)
            if "mask" in output_dict:
                mask_coverage = output_dict["mask"].float().mean().item()
                log_dict["mask_coverage"] = mask_coverage
            
            # Log images side by side if they have correct shape and it's time to log
            should_log_images = step % self.log_every_n_steps == 0
            
            def collect_images(tensor, name_prefix):
                """Collect images from tensor and return as list of (image, label) tuples"""
                images = []
                if len(tensor.shape) >= 2:  # Should have at least batch and feature dimensions
                    # Take first sample from batch for logging
                    img = tensor[0].detach().cpu()
                    
                    # Scale to [0, 1] range for proper visualization (diffusion models often use [-1, 1])
                    img = (img + 1.0) / 2.0  # Scale from [-1, 1] to [0, 1]
                    img = torch.clamp(img, 0.0, 1.0)  # Ensure values are in valid range
                    
                    # Try to determine if this is X or Y part based on size
                    if img.numel() == self.total_X_size and len(self.x_shape) >= 2:
                        # Reshape to X shape and log
                        if len(self.x_shape) == 3:  # (C, H, W)
                            img_reshaped = img.reshape(self.x_shape)
                            if self.x_shape[0] == 1:  # Grayscale
                                images.append((img_reshaped.squeeze(0).numpy(), f"{name_prefix}_x_grayscale"))
                            elif self.x_shape[0] == 3:  # RGB
                                images.append((img_reshaped.permute(1, 2, 0).numpy(), f"{name_prefix}_x_rgb"))
                        elif len(self.x_shape) == 2:  # (H, W) - grayscale
                            img_reshaped = img.reshape(self.x_shape)
                            images.append((img_reshaped.numpy(), f"{name_prefix}_x_grayscale"))
                    elif img.numel() == self.total_Y_size and len(self.y_shape) >= 2:
                        # Reshape to Y shape and log
                        if len(self.y_shape) == 3:  # (C, H, W)
                            img_reshaped = img.reshape(self.y_shape)
                            if self.y_shape[0] == 1:  # Grayscale
                                images.append((img_reshaped.squeeze(0).numpy(), f"{name_prefix}_y_grayscale"))
                            elif self.y_shape[0] == 3:  # RGB
                                images.append((img_reshaped.permute(1, 2, 0).numpy(), f"{name_prefix}_y_rgb"))
                        elif len(self.y_shape) == 2:  # (H, W) - grayscale
                            img_reshaped = img.reshape(self.y_shape)
                            images.append((img_reshaped.numpy(), f"{name_prefix}_y_grayscale"))
                return images
            
            if should_log_images:
                # Separate X and Y images
                x_images = []
                y_images = []
                
                # Collect all available images and separate by X/Y
                if "batch" in output_dict:
                    batch = output_dict["batch"]
                    if len(batch.shape) >= 2:
                        x_part = batch[:, :self.total_X_size]
                        y_part = batch[:, self.total_X_size:]
                        x_images.extend(collect_images(x_part, "original"))
                        y_images.extend(collect_images(y_part, "original"))
                
                if "noisy" in output_dict:
                    noisy = output_dict["noisy"]
                    if len(noisy.shape) >= 2:
                        x_part = noisy[:, :self.total_X_size]
                        y_part = noisy[:, self.total_X_size:]
                        x_images.extend(collect_images(x_part, "noisy"))
                        y_images.extend(collect_images(y_part, "noisy"))
                
                if "denoised" in output_dict:
                    denoised = output_dict["denoised"]
                    if len(denoised.shape) >= 2:
                        x_part = denoised[:, :self.total_X_size]
                        y_part = denoised[:, self.total_X_size:]
                        x_images.extend(collect_images(x_part, "denoised"))
                        y_images.extend(collect_images(y_part, "denoised"))
                
                def create_side_by_side_image(images, variable_name):
                    """Create a simple side-by-side concatenated image"""
                    if not images:
                        return None, None
                    
                    try:
                        # Convert all images to grayscale and get max height
                        processed_images = []
                        max_height = 0
                        image_types = []
                        
                        for img, label in images:
                            # Convert to grayscale if RGB for consistency
                            if len(img.shape) == 3:
                                img_gray = np.mean(img, axis=2)
                            else:
                                img_gray = img
                            processed_images.append(img_gray)
                            # Extract the type (original, noisy, denoised)
                            image_type = label.split('_')[0]  # Gets 'original', 'noisy', or 'denoised'
                            image_types.append(image_type)
                            max_height = max(max_height, img_gray.shape[0])
                        
                        # Pad images to same height
                        padded_images = []
                        for img in processed_images:
                            if img.shape[0] < max_height:
                                padding = max_height - img.shape[0]
                                img = np.pad(img, ((0, padding), (0, 0)), mode='constant', constant_values=0)
                            padded_images.append(img)
                        
                        # Add thin separators between images
                        final_images = []
                        for i, img in enumerate(padded_images):
                            final_images.append(img)
                            if i < len(padded_images) - 1:  # Don't add separator after last image
                                separator = np.ones((max_height, 2)) * 0.5  # 2-pixel wide gray separator
                                final_images.append(separator)
                        
                        # Concatenate horizontally
                        combined_image = np.concatenate(final_images, axis=1)
                        
                        # Create description
                        description = f"{variable_name} variable: " + " → ".join(image_types)
                        
                        return combined_image, description
                    
                    except Exception as e:
                        # Fallback: return first image with error description
                        if images:
                            return images[0][0], f"{variable_name} variable (concatenation failed): {images[0][1]}"
                        return None, None
                
                # Generate and collect samples if enabled
                gen_x_images = []
                gen_y_images = []
                if self.log_generated_samples:
                    try:
                        if self.save_generation_process:
                            # Generate samples with intermediate steps
                            generated_samples, intermediate_steps = self.sample(
                                self.n_generated_samples, self.generation_steps, return_intermediates=True
                            )
                            
                            # Log generation process
                            gen_process_images = []
                            for step_name, samples in intermediate_steps.items():
                                # Take first sample for visualization
                                sample_x = samples[0, :self.total_X_size].detach().cpu()
                                gen_process_images.extend(collect_images(sample_x.unsqueeze(0), step_name))
                            
                            if gen_process_images:
                                gen_process_combined, _ = create_side_by_side_image(gen_process_images, "Generation Process")
                                if gen_process_combined is not None:
                                    log_dict["generation_process"] = wandb.Image(gen_process_combined, caption="Generation process steps")
                        else:
                            # Just generate final samples
                            generated_samples = self.sample(self.n_generated_samples, self.generation_steps)
                        
                        # Convert generated samples to the same format as other images
                        gen_x = generated_samples[:, :self.total_X_size].detach().cpu()
                        gen_y = generated_samples[:, self.total_X_size:].detach().cpu()
                        
                        gen_x_images.extend(collect_images(gen_x, "generated"))
                        gen_y_images.extend(collect_images(gen_y, "generated"))
                    except Exception as e:
                        print(f"Error generating samples for logging: {e}")
                
                # Create and log training process images (original, noisy, denoised)
                if x_images:
                    x_combined, x_description = create_side_by_side_image(x_images, "Training X")
                    if x_combined is not None:
                        log_dict["training_process_X"] = wandb.Image(x_combined, caption=x_description)
                
                if y_images:
                    y_combined, y_description = create_side_by_side_image(y_images, "Training Y")
                    if y_combined is not None:
                        log_dict["training_process_Y"] = wandb.Image(y_combined, caption=y_description)
                
                # Create and log generated samples separately
                if gen_x_images:
                    gen_x_combined, gen_x_description = create_side_by_side_image(gen_x_images, "Generated X")
                    if gen_x_combined is not None:
                        log_dict["generated_samples_X"] = wandb.Image(gen_x_combined, caption=gen_x_description)
                
                if gen_y_images:
                    gen_y_combined, gen_y_description = create_side_by_side_image(gen_y_images, "Generated Y")
                    if gen_y_combined is not None:
                        log_dict["generated_samples_Y"] = wandb.Image(gen_y_combined, caption=gen_y_description)
            
            wandb.log(log_dict)
    
    def log_mean_coeff(self, t):
        return -0.25 * t ** 2 * (self.diffusion_config["beta_max"] - self.diffusion_config["beta_min"]) - 0.5 * t * self.diffusion_config["beta_min"]

    def beta_t(self, t):
        return self.diffusion_config["beta_min"] + t * (self.diffusion_config["beta_max"] - self.diffusion_config["beta_min"])

    def sample(self, n_samples, n_steps=100, return_intermediates=False):
        """
        Sample from the model using reverse diffusion process.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        n_steps : int, optional
            Number of sampling steps (default: 100).
        return_intermediates : bool, optional
            Whether to return intermediate steps (default: False).
        
        Returns
        -------
        torch.Tensor or tuple
            Generated samples (x, y), and optionally intermediate steps.
        """
        self.backbone.eval()
        
        intermediate_steps = {} if return_intermediates else None
        
        with torch.no_grad():
            # Start from pure noise at t=1
            total_dim = self.total_X_size + self.total_Y_size
            xy_t = torch.randn(n_samples, total_dim, device=self.device)
            
            # Save initial noise if tracking intermediates
            if return_intermediates:
                intermediate_steps["step_000_noise"] = xy_t.clone()
            
            # Time steps for reverse process
            dt = - 1.0 / n_steps
            dt = dt * torch.ones(n_samples, 1, device=self.device)

            for i in range(n_steps):
                t = 1 - i/n_steps
                t = t * torch.ones(n_samples, 1, device=self.device)
                log_mean_coeff = self.log_mean_coeff(t)
                std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
                beta_t = self.beta_t(t)
                drift = -0.5 * beta_t * xy_t

                diffusion = torch.sqrt(beta_t)
                marginal_mask_index_vector = torch.zeros(n_samples, dtype=torch.float32, device=self.device)
                cond = torch.cat([t, marginal_mask_index_vector.unsqueeze(1)], dim=1).to(self.device)
                
                if self.variant == "c":
                    batch_mask = self.get_masks(marginal_mask_index_vector)
                    xy_t = self.apply_masks_to_data(xy_t, torch.zeros_like(xy_t), batch_mask)
                
                if self.use_ema:
                    score = self.ema.module(xy_t, cond)
                else:
                    score = self.backbone(xy_t, cond)
                score = score.reshape(n_samples, -1)

                score = -score / std
                
                drift = drift - (diffusion**2) * score

                # assert not torch.isnan(score).any(), "B) Score contains NaN values"

                xy_t_mean = xy_t + drift * dt

                noise = torch.randn_like(xy_t, device=self.device)
                xy_t = xy_t_mean + diffusion * torch.sqrt(-dt) * noise
                # assert not torch.isnan(torch.sqrt(dt)).any(), f"torch.sqrt(dt) contains NaN values at step {i}"
                
                # Save intermediate steps (every 20% of the process)
                if return_intermediates and i % max(1, n_steps // 5) == 0 and i > 0:
                    progress = int(i / n_steps * 100)
                    intermediate_steps[f"step_{i:03d}_{progress:02d}%"] = xy_t.clone()

        self.backbone.train()
        
        if return_intermediates:
            # Save final result
            intermediate_steps["step_final"] = xy_t_mean.clone()
            return xy_t_mean, intermediate_steps
        else:
            return xy_t_mean
    
    def get_mutual_information(
        self,
        dataloader,
        device,
    ) -> float:
        """
        Mutual information estimation.
        
        Parameters
        ----------
        dataloader
            Data loader. Must yield tuples (x,y,z).
        loss : callable
            Mutual information neural estimation loss.
        device
            Comoutation device.
        marginalize : str, optional
            Method of marginalizing the joint distribution.
            Is either "permute" or "product".
        """
        
        self.backbone.eval()
        
        sum_mi = 0.0
        sum_mi_sigma = 0.0
        total_elements = 0
        
        with torch.no_grad():
            for index, batch in enumerate(dataloader):
                x, y = batch
                batch_size = x.shape[0]

                x, y = x.to(device), y.to(device)

                x = x.reshape(batch_size, -1)
                y = y.reshape(batch_size, -1)

                batch_size = x.shape[0]

                t = (sample_importance_sampling(
                        shape=(batch_size, 1), beta_min=self.diffusion_config["beta_min"], beta_max=self.diffusion_config["beta_max"], T=1.0)).to(self.device)
                xy_0 = torch.cat((x, y), dim=1)
                log_mean_coeff = -0.25 * t ** 2 * \
                    (self.diffusion_config["beta_max"] - self.diffusion_config["beta_min"]) - 0.5 * t * self.diffusion_config["beta_min"]
                mean = torch.exp(log_mean_coeff)
                std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
                noise = torch.randn_like(xy_0, device=self.device)
                xy_t = xy_0 * mean + noise * std

                scores = self.get_all_scores(xy_0, xy_t, t)

                beta_t = self.diffusion_config["beta_min"] + t * (self.diffusion_config["beta_max"] - self.diffusion_config["beta_min"])

                mi, mi_sigma = self.compute_mi(
                    scores,
                    xy_t,
                    torch.sqrt(beta_t),
                    mean,
                    std,
                    self.sigma
                )

                sum_mi += mi
                sum_mi_sigma += mi_sigma

                total_elements += batch_size

        self.backbone.train()

        mi = sum_mi / total_elements
        mi = max(mi, 0.0)
        mi_sigma = sum_mi_sigma / total_elements
        mi_sigma = max(mi_sigma, 0.0)
        
        return mi, mi_sigma
    
    def compute_mi(self, scores, xy_t, g, mean, std, sigma):
        """
        Compute mutual information from scores, noisy data and parameters.
        
        Parameters
        ----------
        scores : torch.Tensor
            Scores computed by the backbone.
        xy_t : torch.Tensor
            Noisy data at time t.
        g : torch.Tensor
            Importance sampling weights.
        mean : torch.Tensor
            Mean of the diffusion process.
        std : torch.Tensor
            Standard deviation of the diffusion process.
        sigma : torch.Tensor
            Sigma parameter for the diffusion process.
        
        Returns
        -------
        float
            Estimated mutual information.
        """
        if self.variant == "j":
            mi = self.mi_joint(*scores, g)
            mi_sigma = self.mi_joint_sigma(*scores, xy_t, mean, std, g, sigma)
        elif self.variant == "c":
            mi = self.mi_cond(*scores, g)
            mi_sigma = self.mi_cond_sigma(*scores, xy_t, g, mean, std, sigma)
        else:
            raise NotImplementedError("Unknown variant: {}".format(self.variant))
        return mi, mi_sigma

    def mi_cond_sigma(self,s_marg ,s_cond, xy_t, g, mean ,std,sigma):

        x_t = xy_t[:, :self.total_X_size]
        M = g.shape[0] 

        s_marg = s_marg.view(M,-1)
        s_cond = s_cond.view(M,-1)
        
        chi_t_x = mean**2 * sigma**2 + std**2
        ref_score_x = -(x_t)/chi_t_x
        ref_score_x = ref_score_x.view(M,-1)
        const = get_normalizing_constant((1,),beta_min=self.diffusion_config["beta_min"],beta_max=self.diffusion_config["beta_max"]).to(s_marg.device)
        e_marg = - const * 0.5 * ((s_marg - std* ref_score_x)**2).sum() 
        e_cond = - const * 0.5 * ((s_cond - std* ref_score_x)**2).sum() 
        return (e_marg - e_cond).item()

    def mi_cond(self,s_marg ,s_cond, g):

        M = g.shape[0] 
        const = get_normalizing_constant((1,),beta_min=self.diffusion_config["beta_min"],beta_max=self.diffusion_config["beta_max"]).to(s_marg.device)

        s_marg = s_marg.view(M,-1)
        s_cond = s_cond.view(M,-1)

        mi = const *0.5* ((s_marg - s_cond  )**2).sum()
        return mi.item()



    def mi_joint(self, s_joint ,s_cond_x, s_cond_y, g ):
    
        
        M = g.shape[0] 
        s_cond = torch.cat([s_cond_x,s_cond_y],dim=1)
        s_cond = s_cond.view(M,-1)
        s_joint = s_joint.view(M,-1)

        const = get_normalizing_constant((1,),beta_min=self.diffusion_config["beta_min"],beta_max=self.diffusion_config["beta_max"]).to(g.device)
        try:
            mi = const *0.5* ((s_joint - s_cond  )**2).sum()
        except:
            mshapes = list(m.shape for m in self.masks)
            raise ValueError(f"Shapes - s_joint: {s_joint.shape}, s_cond: {s_cond.shape}, g: {g.shape}, const: {const.shape},\
                             masks shape: {mshapes}")
        print(f"mi item is {mi.item()}")  # Debugging line to check MI value --- IGNORE ---
        return mi.item()


    def mi_joint_sigma(self, s_joint ,s_cond_x,s_cond_y,xy_t,mean,std, g ,sigma):
        
        M = g.shape[0] 

        s_cond_x = s_cond_x.view(M,-1)
        s_cond_y = s_cond_y.view(M,-1)
        s_joint = s_joint.view(M,-1)

        x_t = xy_t[:, :self.total_X_size]
        y_t = xy_t[:, self.total_X_size:]
        
        chi_t = mean**2 * sigma**2 + std**2
        ref_score_x = -(x_t)/chi_t 
        ref_score_y = -(y_t)/chi_t 

        ref_score_x = ref_score_x.view(M,-1)
        ref_score_y = ref_score_y.view(M,-1)

        ref_score_xy =  torch.cat([ref_score_x,ref_score_y],dim=1)

        const = get_normalizing_constant((1,),beta_min=self.diffusion_config["beta_min"],beta_max=self.diffusion_config["beta_max"]).to(g.device)
        e_joint = -const *0.5* ((s_joint - std * ref_score_xy  )**2).sum()
        e_cond_x = -const *0.5* ((s_cond_x - std * ref_score_x  )**2).sum()
        e_cond_y = -const *0.5* ((s_cond_y - std * ref_score_y  )**2).sum()
            
        return (e_joint - e_cond_x - e_cond_y  ).item()
    
    @property
    def masks(self):
        if self.variant == "j":
            masks = [
                torch.ones(self.total_X_size + self.total_Y_size, dtype=torch.float32),
                torch.cat([torch.ones(self.total_X_size, dtype=torch.float32), 
                           -torch.ones(self.total_Y_size, dtype=torch.float32)]),
                torch.cat([-torch.ones(self.total_X_size, dtype=torch.float32),
                            torch.ones(self.total_Y_size, dtype=torch.float32)]),
            ]
        elif self.variant == "c":
            masks = [
                torch.cat([torch.ones(self.total_X_size, dtype=torch.float32), 
                           torch.zeros(self.total_Y_size, dtype=torch.float32)]),
                torch.cat([torch.ones(self.total_X_size, dtype=torch.float32),
                            -torch.ones(self.total_Y_size, dtype=torch.float32)]),
            ]
        masks = torch.vstack(masks)
        return masks
    
    def get_random_mask_index(self, batch_size):
        """
        Get random mask index for the batch.
        
        Parameters
        ----------
        batch_size : int
            Size of the batch.
        
        Returns
        -------
        torch.Tensor
            Random mask index.
        """
        if self.variant == "j":
            # Joint variant
            return torch.randint(0,3,(batch_size,)).to(self.device)
        elif self.variant == "c":
            return torch.randint(0,2,(batch_size,)).to(self.device)
        else:
            raise NotImplementedError("Unknown variant: {}".format(self.variant))

    def apply_masks_to_data(self, diffused_data, original_data, batch_mask):
        """
        Apply masks to the tensors.
        
        Parameters
        ----------
        diffused_data : torch.Tensor
            Diffused batch of data.
        original_data : torch.Tensor
            Original batch of data.
        batch_mask : 1-D torch.Tensor
            Masks for the batch.

        Returns
        -------
        torch.Tensor
            Masked data.
        """
        try:
            masked_data = batch_mask * diffused_data
        except:
            raise ValueError(f"Batch mask shape: {batch_mask.shape}, Diffused data shape: {diffused_data.shape}")
        masked_data = torch.where(batch_mask < 0, original_data, masked_data)

        return masked_data

    def get_masks(self, mask_indices_vector):
        masks = self.masks.unsqueeze(0).expand(mask_indices_vector.shape[0], -1, -1).to(mask_indices_vector.device)
        mask_indices_vector = mask_indices_vector[:, None, None].long()
        mask_indices_vector = mask_indices_vector.expand(-1, -1, masks.shape[2])
        # Gather the masks for the batch
        batch_mask = torch.gather(masks, 1, mask_indices_vector).squeeze(1)
        return batch_mask

    def get_all_scores(self, xy_0, xy_t, t):
        """
        Get scores for all the possible masks.
        
        Parameters
        ----------
        xy_0 : torch.Tensor
            Original data.
        xy_t : torch.Tensor
            Noisy data at time t.
        t : torch.Tensor
            Time tensor.
        
        Returns
        -------
        tuple of torch.Tensor
            Scores from the backbone.
        """

        batch_size = xy_t.shape[0]
        scores = []

        log_mean_coeff = -0.25 * t ** 2 * \
            (self.diffusion_config["beta_max"] - self.diffusion_config["beta_min"]) - 0.5 * t * self.diffusion_config["beta_min"]
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        
        for mask_idx in range(len(self.masks)):
            mask_indices_vector = mask_idx * torch.ones(batch_size, dtype=torch.float32).to(self.device)
            batch_mask = self.get_masks(mask_indices_vector)
            xy_masked = self.apply_masks_to_data(xy_t, xy_0, batch_mask)
            cond = torch.cat([t, mask_indices_vector.unsqueeze(1)], dim=1).to(self.device)
            if self.use_ema:
                score = self.ema.module(xy_masked, cond)
            else:
                score = self.backbone(xy_masked, cond)
            score = score.reshape(batch_size, -1)

            x_denoised = (xy_t - std * score) / mean
            x_denoised = self.apply_masks_to_data(x_denoised, xy_0, batch_mask)
            has_data_been_diffused = batch_mask > 0  # Only where mask>0 the data is being diffused and that part used for the loss
            score = score[has_data_been_diffused].reshape(batch_size, -1)
            scores.append(score)

        return tuple(scores)

    def loss(self, x, y):

        batch_size = x.shape[0]

        x = x.reshape(batch_size, -1)
        y = y.reshape(batch_size, -1)

        t = (sample_importance_sampling(
            shape=(batch_size, 1), beta_min=self.diffusion_config["beta_min"], beta_max=self.diffusion_config["beta_max"], T=1.0)).to(self.device)
        xy_0 = torch.cat((x, y), dim=1)
        log_mean_coeff = -0.25 * t ** 2 * \
            (self.diffusion_config["beta_max"] - self.diffusion_config["beta_min"]) - 0.5 * t * self.diffusion_config["beta_min"]
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        noise = torch.randn_like(xy_0, device=self.device)
        try:
            xy_t = xy_0 * mean + noise * std
        except:
            raise ValueError(f"Shapes - xy_0: {xy_0.shape}, mean: {mean.shape}, noise: {noise.shape}, std: {std.shape}")
        random_mask_indeces = self.get_random_mask_index(batch_size)
        batch_mask = self.get_masks(random_mask_indeces)
        xy_masked = self.apply_masks_to_data(xy_t, xy_0, batch_mask)
        cond = torch.cat([t, random_mask_indeces.unsqueeze(1)], dim=1).to(self.device)

        score = self.backbone(xy_masked, cond).reshape(batch_size, -1)
        try:
            xy_denoised = (xy_t - std * score) / mean
        except:
            raise ValueError(f"Shapes - xy_t: {xy_t.shape}, std: {std.shape}, score: {score.shape}, mean: {mean.shape}")
        xy_denoised = self.apply_masks_to_data(xy_denoised, xy_0, batch_mask)
        has_data_been_diffused = batch_mask > 0  # Only where mask>0 the data is being diffused and that part used for the loss
        torch.where(has_data_been_diffused, score, torch.zeros_like(score))
        torch.where(has_data_been_diffused, noise, torch.zeros_like(noise))
        
        noise = noise.reshape(batch_size, -1)
        has_data_been_diffused = has_data_been_diffused.reshape(batch_size, -1)

        number_of_diffused_elements = torch.sum(has_data_been_diffused, dim=1, keepdim=True)
        total_number_of_elements = has_data_been_diffused.shape[1]

        loss_weight = number_of_diffused_elements / total_number_of_elements
        loss = torch.sum(
            has_data_been_diffused * (score - noise) ** 2, dim=1
        ) / number_of_diffused_elements

        loss = loss * loss_weight

        output_dict = {
            "denoised": xy_denoised,
            "score": score,
            "noise": noise,
            "loss": loss,
            "noisy": xy_t,
            "t": t,
            "mask": has_data_been_diffused,
            "batch": xy_0
        }

        return loss, output_dict

class EMA(torch.nn.Module):
    """ Model Exponential Moving Average V2 from timm"""
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class GenericConv2DDiffuser(torch.nn.Module):

    def __init__(
        self,
        X_shape: tuple,
        Y_shape: tuple,
        n_filters: int=16,
        hidden_dim: int=128,
        n_X_convolutions: int=None,
        n_Y_convolutions: int=None,
    ) -> None:
        super().__init__()

        if (not len(X_shape) in [3, 4]) or (not len(Y_shape) in [3, 4]):
            raise ValueError("Inputs shpuld be batches of images.")

        if (X_shape[-2] != X_shape[-1]) or (Y_shape[-2] != Y_shape[-1]):
            raise ValueError("Input images have to be square.")

        n_X_channels = X_shape[1] if (len(X_shape) == 4) else 1
        n_Y_channels = Y_shape[1] if (len(Y_shape) == 4) else 1

        self.total_X_size = X_shape[-1] * X_shape[-2]
        self.total_Y_size = Y_shape[-1] * Y_shape[-2]

        self.X_shape = X_shape
        self.Y_shape = Y_shape

        self.n_filters = n_filters

        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(2, 2*n_filters),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2*n_filters, 2*n_filters)
        )
        
        # Convolution layers.
        # TODO: reuse code!
        if n_X_convolutions is None:
            log2_remaining_size = 2
            n_X_convolutions = int(math.floor(math.log2(X_shape[-1]))) - log2_remaining_size
            
        self.X_convolutions = torch.nn.ModuleList([torch.nn.Conv2d(n_X_channels, n_filters, kernel_size=3, padding='same')] + \
                [torch.nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='same') for index in range(n_X_convolutions - 1)])
        for conv_index in range(n_X_convolutions):
            X_shape = X_shape[:-2] + ((X_shape[-2] - 2) // 2 + 1, (X_shape[-1] - 2) // 2 + 1,)

        if n_Y_convolutions is None:
            log2_remaining_size = 2
            n_Y_convolutions = int(math.floor(math.log2(Y_shape[-1]))) - log2_remaining_size
        for conv_index in range(n_Y_convolutions):
            Y_shape = Y_shape[:-2] + ((Y_shape[-2] - 2) // 2 + 1, (Y_shape[-1] - 2) // 2 + 1,)
            
        self.Y_convolutions = torch.nn.ModuleList([torch.nn.Conv2d(n_Y_channels, n_filters, kernel_size=3, padding='same')] + \
                [torch.nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='same') for index in range(n_Y_convolutions - 1)])

        self.activation = torch.nn.LeakyReLU()
        self.maxpool2d = torch.nn.MaxPool2d((2,2))

        # Dense part.
        remaining_dim_X = n_filters * X_shape[-1] * X_shape[-2]
        remaining_dim_Y = n_filters * Y_shape[-1] * Y_shape[-2]
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(remaining_dim_X + remaining_dim_Y, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, self.total_X_size + self.total_Y_size),
        )
    
    def forward(self, xy: torch.Tensor, cond: torch.Tensor) -> torch.tensor:

        x = xy[:, :self.total_X_size].reshape(-1, *self.X_shape)
        y = xy[:, self.total_X_size:].reshape(-1, *self.Y_shape)

        t_xy = self.time_mlp(cond)

        t_x = t_xy[:, :self.n_filters].reshape(-1, self.n_filters, 1, 1)
        t_y = t_xy[:, self.n_filters:].reshape(-1, self.n_filters, 1, 1)

        # Convolution layers.
        for conv2d in self.X_convolutions:
            x = conv2d(x)
            try:
                x = x + t_x
            except:
                raise ValueError(f"Shapes - x: {x.shape}, t_x: {t_x.shape}, conv2d: {conv2d}")
            x = self.maxpool2d(x)
            x = self.activation(x)
            
        for conv2d in self.Y_convolutions:
            y = conv2d(y)
            y = y + t_y
            y = self.maxpool2d(y)
            y = self.activation(y)

        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        
        xy = self.dense(torch.cat((x, y), dim=1))
        x = xy[:, :self.total_X_size].reshape(-1, *self.X_shape)
        y = xy[:, self.total_X_size:].reshape(-1, *self.Y_shape)

        xy = torch.cat((x, y), dim=1)
        return xy

class GenericMLPDiffuser(torch.nn.Module):

    def __init__(
        self,
        X_shape: tuple,
        Y_shape: tuple,
        hidden_dim: int=128,
    ) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(X_shape[-1] + Y_shape[-1], hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, X_shape[-1] + Y_shape[-1])
        self.act = torch.nn.LeakyReLU()

        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, xy: torch.Tensor, cond: torch.Tensor) -> torch.tensor:
        t_xy = self.time_mlp(cond)
        xy = self.linear1(xy)
        xy = xy + t_xy
        xy = self.act(xy)
        xy = self.linear2(xy)
        xy = xy + t_xy
        xy = self.act(xy)
        xy = self.linear3(xy)
        return xy

from diffusers import UNet2DModel

# UNet backbone for diffusion
class UNet(torch.nn.Module):
    def __init__(self, X_shape, Y_shape, hidden_dim=64, nb_var=2, norm_num_groups=8, name="mod"):
        super(UNet, self).__init__()
        self.name =name
        self.X_shape = X_shape
        self.Y_shape = Y_shape
        self.total_X_size = X_shape[-1] * X_shape[-2]
        self.total_Y_size = Y_shape[-1] * Y_shape[-2]
        # UNet from diffusers library
        sample_size = X_shape[-1]
        channels_x = X_shape[0]
        channels_y = Y_shape[0]
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=channels_x + channels_y,
            out_channels=channels_x + channels_y,
            layers_per_block=2,
            block_out_channels=(hidden_dim, hidden_dim * 2, hidden_dim * 4),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
            norm_num_groups=norm_num_groups,
        )
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(nb_var, 1),
            torch.nn.GELU(),
            torch.nn.Linear(1, 1)
        )


    def forward(self, x, timestep, condition=None):
        # Pass the latent through the U-Net for denoising
        x = x.reshape(-1, self.X_shape[0]+self.Y_shape[0], *self.X_shape[-2:])
        timestep = self.time_mlp(timestep).squeeze(-1)
        ret = self.unet(x, timestep)["sample"]
        return ret