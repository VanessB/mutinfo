from ..tools import mapped_multi_rv_frozen
from .ode_utils import get_ode_sampler
from .drifts import *
import torch
from torch.utils.data import TensorDataset, DataLoader


class GenerativeRV(mapped_multi_rv_frozen):

    def __init__(self, base_rv, ode_sampler_dict=None, hf_path=None, batch_size=64, device='cuda', step_size=1e-2, method='euler', eps=1e-5, *args, **kwargs):

        assert ode_sampler_dict is not None or hf_path is not None, f"Either a path to a hf model or ode_sampler_dict must be provided"

        if ode_sampler_dict is not None:
            shape = ode_sampler_dict["shape"]
            ode_sampler = get_ode_sampler(ode_sampler_dict["drift_fn"], ode_sampler_dict["inverse_scaler"], step_size=step_size, method=method, device=device, eps=eps)
        else:
            drift_fn, scaler_fn, shape = auto_drift_from_hf(hf_path, step_size, device=device)
            ode_sampler = get_ode_sampler(drift_fn, scaler_fn, step_size=step_size, method=method, device=device, eps=eps)
        ode_mapping = self.get_ode_mapping(ode_sampler, batch_size)
        self.num_features = int(np.prod(shape))
        self.shape = shape
        self.device = device

        super().__init__(base_rv, ode_mapping, ode_mapping, *args, **kwargs)
    
    def get_ode_mapping(self, ode_sampler, batch_size):
        def ode_map(x,y):
            
            x = torch.tensor(x, device=self.device)
            y = torch.tensor(y, device=self.device)

            if x.shape[1] < self.num_features:
                x_noise = torch.randn(x.shape[0], self.num_features - x.shape[1], device=self.device)*torch.std(x)+torch.mean(x)
                x = torch.cat((x, x_noise), dim=1)
                x = x.reshape(x.shape[0], *(self.shape))
            
            if y.shape[1] < self.num_features:
                y_noise = torch.randn(y.shape[0], self.num_features - y.shape[1], device=self.device)*torch.std(y)+torch.mean(y)
                y = torch.cat((y, y_noise), dim=1)
                y = y.reshape(y.shape[0], *(self.shape))
            
            x_mapped = torch.empty(0, *self.shape).to(x)
            y_mapped = torch.empty(0, *self.shape).to(y)
            dataset = TensorDataset(x, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for x_batch, y_batch in dataloader:
                x_batch_mapped = ode_sampler(x_batch)
                y_batch_mapped = ode_sampler(y_batch)
                x_mapped = torch.cat((x_mapped, x_batch_mapped), dim=0)
                y_mapped = torch.cat((y_mapped, y_batch_mapped), dim=0)
            return x_mapped.cpu().numpy(), y_mapped.cpu().numpy()
        return ode_map


def generate_dataset(rv, path, size):
    x, y = rv.rvs(size=size)
    np.savez_compressed(path, x=x.cpu().numpy(), y=y.cpu().numpy())