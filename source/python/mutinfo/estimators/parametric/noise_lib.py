import abc
import torch
import torch.nn as nn
import numpy as np


class Noise(abc.ABC, nn.Module):
    """
    Baseline forward method to get the total + rate of noise at a timestep
    """
    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

    """
    Assume time goes from 0 to 1
    """
    @abc.abstractmethod
    def rate_noise(self, t):
        """
        Rate of change of noise ie g(t)
        """
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        """
        Total noise ie \int_0^t g(t) dt + g(0)
        """
        pass


class GeometricNoise(Noise, nn.Module):
    def __init__(self, sigma_min=1e-3, sigma_max=1, learnable=False):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])
        if learnable:
            self.sigmas = nn.Parameter(self.sigmas)
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (self.sigmas[1].log() - self.sigmas[0].log())

    def total_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t


class LogLinearNoise(Noise, nn.Module):
    """
    Log Linear noise schedule built so that 1 - 1/e^(n(t)) interpolates between 0 and ~1
    when t goes from 0 to 1. Used for absorbing

    Total noise is -log(1 - (1 - eps) * t), so the sigma will be (1 - eps) * t
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

class CosineSquaredNoise(Noise, nn.Module):
    """
    Cosine squared noise schedule
    """
    def __init__(self):
        super().__init__()
    
    def rate_noise(self, t):
        return -np.pi  * torch.tan(np.pi / 2 * (1-t))
    
    def total_noise(self, t):
        return -torch.log(torch.pow(torch.cos((1-t) * np.pi / 2), 2))

class CosineNoise(Noise, nn.Module):
    """
    Cosine noise schedule
    """
    def __init__(self):
        super().__init__()
    
    def total_noise(self, t):
        return -torch.log(torch.cos((1-t) * np.pi / 2))
    
    def rate_noise(self, t):
        return -np.pi / 2 * torch.tan(np.pi / 2 * (1-t))

class LinearNoise(Noise, nn.Module):
    """
    Linear noise schedule
    """
    def __init__(self, sigma_max=1):
        super().__init__()
        self.empty = nn.Parameter(torch.tensor(0.0))
        self.sigma_max = sigma_max

    def rate_noise(self, t):
        return self.sigma_max

    def total_noise(self, t):
        return self.sigma_max*t

class LearnableNoise(Noise):

  def __init__(self, base_noise, vocab_size):
    super().__init__()
    self.base_noise = base_noise
    self.noise = nn.Parameter(torch.zeros(vocab_size))
    self.vocab_size = vocab_size

  def noise_norm(self):
    # print(f"noise norm: {torch.norm(torch.sigmoid(self.noise), p=2)}")
    return torch.norm(torch.sigmoid(self.noise), p=2)  
  
  def total_noise(self, t):
    assert t.ndim == 2, f't should be a 1D tensor, instead got {t.shape} tensor'
    total_noise = self.base_noise.total_noise(t)
    total_noise = total_noise.expand(t.size(0), self.vocab_size)
    mult_factor = (1+torch.sigmoid(self.noise).unsqueeze(0).expand(t.size(0), self.vocab_size))
    mult_factor[:,-1] = 1
    total_noise = total_noise * mult_factor
    return total_noise
  
  def rate_noise(self, t, eps=1e-3):
    rate_noise = self.total_noise(t+eps) - self.total_noise(t)
    return rate_noise / eps
