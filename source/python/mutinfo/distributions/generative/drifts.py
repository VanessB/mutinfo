import torch
import numpy as np
from diffusers import DiffusionPipeline
from sklearn.datasets import make_moons
from tqdm import tqdm

def auto_drift_from_hf(id, step_size, device='cuda'):
    pipe = DiffusionPipeline.from_pretrained(id)
    pipe_class = str(type(pipe))

    if 'scoresdeve' in pipe_class.lower():
        sigma_min = pipe.scheduler.config.sigma_min
        sigma_max = pipe.scheduler.config.sigma_max
        _model = pipe.unet.to(device).eval()
        model = lambda x, sigma: _model(x, sigma).sample
        def ve_sde_inverse_scaler(x):
            x = x.clamp(0,1)
            return x
        shape = (3, pipe.unet.config.sample_size, pipe.unet.config.sample_size)
        return ve_sde_drift(sigma_min, sigma_max, model, step_size), ve_sde_inverse_scaler, shape
    if 'cfm-cifar10-32' in id:
        _model = pipe.unet.to(device).eval()
        
        def model_wrapper(x, t):
            # Ensure t is in [0, 1] range and map to [0, 999]
            timesteps = (t * 999).long().clamp(0, 999)
            return _model(x, timesteps).sample
        
        def ddpm_inverse_scaler(x):
            x = (x / 2 + 0.5).clamp(0, 1)
            return x
            
        return flow_drift(model_wrapper), ddpm_inverse_scaler, (3, 32, 32)
    raise NotImplementedError(f"{pipe_class} not supported")

def flow_drift(flow_model):

    def drift_fn(x, t):
        return flow_model(x, t)

    return drift_fn

def ve_sde_drift(sigma_min, sigma_max, model, step_size):

    def drift_fn(x, t):
        sigma = sigma_min * (sigma_max/sigma_min)**t
        diffusion = sigma * torch.sqrt(torch.torch.Tensor(2 * (np.log(sigma_max) - np.log(sigma_min)), device=t.device))
        score = model(x, sigma)
        print(t)
        # expand diffusion to same shape as score
        while diffusion.dim() < score.dim():
            diffusion = diffusion.unsqueeze(-1)
        drift = -0.5 * diffusion**2 * score
        return drift
    
    return drift_fn

def flow_make_moons(base_rv, device='cuda'):

    class Flow(torch.nn.Module):
        def __init__(self, dim: int = 2, h: int = 64):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dim + 1, h), torch.nn.ELU(),
                torch.nn.Linear(h, h), torch.nn.ELU(),
                torch.nn.Linear(h, h), torch.nn.ELU(),
                torch.nn.Linear(h, dim))

        def forward(self, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
            while t.ndim < x_t.ndim:
                t = t.unsqueeze(-1)
            return self.net(torch.cat((t, x_t), -1))

        def step(self, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
            t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)

            return x_t + (t_end - t_start) * self(t=t_start + (t_end - t_start) / 2, x_t= x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2)


    def train_flow_model():
            
        flow_x = Flow().to(device)

        optimizer_x = torch.optim.Adam(flow_x.parameters(), 1e-2)
        loss_fn = torch.nn.MSELoss()

        for _ in tqdm(range(10000), desc='Training flow make_moons model'):
            x_0 = torch.Tensor(make_moons(256, noise=0.05)[0]).to(device)
            # x_0 = torch.randn_like(x_1)
            x, _ = base_rv.rvs(size=x_0.shape[0])
            x_1 = torch.Tensor(x).to(device)
            t = torch.rand(len(x_1), 1).to(device)

            x_t = (1 - t) * x_0 + t * x_1
            dx_t = x_1 - x_0

            optimizer_x.zero_grad()
            loss_fn(flow_x(t=t, x_t=x_t), dx_t).backward()
            optimizer_x.step()
        
        return flow_x
    
    flow = train_flow_model()

    class drift_fn(torch.nn.Module):
        def __init__(self, flow):
            super().__init__()
            self.flow = flow

        def forward(self, x, t):
            flow = self.flow.to(x.device)
            return flow(t, x)

    return drift_fn(flow)

def cifar10_drift():
    return auto_drift_from_hf("FrankCCCCC/cfm-cifar10-32", step_size=1e-2)
