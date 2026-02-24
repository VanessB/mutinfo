import torch

class FlowMLP(torch.nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + 1, h), torch.nn.ELU(),
            torch.nn.Linear(h, h), torch.nn.ELU(),
            torch.nn.Linear(h, h), torch.nn.ELU(),
            torch.nn.Linear(h, dim))

    def forward(self, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((t, x_t), -1))