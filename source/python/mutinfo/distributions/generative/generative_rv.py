from ..tools import mapped_multi_rv_frozen
from .ode_utils import get_ode_sampler
from .drifts import *
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


class Generative(mapped_multi_rv_frozen):

    def __init__(self, base_rv, generative_model, batch_size=256, shape=(1, 28, 28), verbose=False, *args, **kwargs):

        self.base_rv = base_rv
        self.generative_model = generative_model
        self.batch_size = batch_size
        self.shape = shape
        self.device = next(generative_model.parameters()).device
        self.verbose = verbose
        ode_mapping = self.get_ode_mapping()

        super().__init__(base_rv, ode_mapping, ode_mapping, *args, **kwargs)
    
    def get_ode_mapping(self):
        def ode_map(x,y):
            
            x = torch.tensor(x, device=self.device)
            y = torch.tensor(y, device=self.device)
            
            x_mapped = None
            y_mapped = None
            dataset = TensorDataset(x, y)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            if self.verbose:
                dataloader = tqdm(dataloader, desc='Mapping samples through generative model')
            for x_batch, y_batch in dataloader:
                x_batch_mapped = self.generative_model.generate_samples(prior_samples=x_batch.float())
                y_batch_mapped = self.generative_model.generate_samples(prior_samples=y_batch.float())

                if self.shape is not None:
                    x_batch_mapped = x_batch_mapped.view(-1, *self.shape)
                    y_batch_mapped = y_batch_mapped.view(-1, *self.shape)

                if x_mapped is None:
                    x_mapped = x_batch_mapped
                    y_mapped = y_batch_mapped
                    continue
                x_mapped = torch.cat((x_mapped, x_batch_mapped), dim=0)
                y_mapped = torch.cat((y_mapped, y_batch_mapped), dim=0)
            return x_mapped.cpu().numpy(), y_mapped.cpu().numpy()
        return ode_map


def generate_dataset(rv, path, size):
    x, y = rv.rvs(size=size)
    np.savez_compressed(path, x=x.cpu().numpy(), y=y.cpu().numpy())