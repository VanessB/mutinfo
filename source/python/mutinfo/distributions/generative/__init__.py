"""Generative models based on flow matching."""

from .lightning_flow import FlowLightningModule
from .datamodule import MNISTFlowDataModule, CustomFlowDataModule
from .ema import EMA
from .flow_MLP import FlowMLP
from .generative_rv import Generative


__all__ = [
    "FlowLightningModule",
    "MNISTFlowDataModule", 
    "CustomFlowDataModule",
    "EMA",
    "FlowMLP",
    "Generative",
]

def generative(base_rv, generative_model, batch_size=256, *args, **kwargs):
    return Generative(base_rv, generative_model, batch_size=batch_size, *args, **kwargs)
