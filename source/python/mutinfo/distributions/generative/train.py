"""Training script for flow-based generative models using PyTorch Lightning and Hydra."""

import sys
from pathlib import Path

# Add the source/python directory to the path so mutinfo can be imported
source_python = Path(__file__).resolve().parents[3]
if str(source_python) not in sys.path:
    sys.path.insert(0, str(source_python))

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="train")
def train(cfg: DictConfig):
    """
    Main training function.
    
    Args:
        cfg: Hydra configuration
    """
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed for reproducibility
    if 'seed' in cfg:
        import pytorch_lightning as pl
        pl.seed_everything(cfg.seed)
    
    # Instantiate datamodule
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    # Instantiate model
    model = hydra.utils.instantiate(cfg.lightning_module)
    
    # Instantiate trainer (with callbacks and logger)
    trainer = hydra.utils.instantiate(cfg.trainer_full)
    
    # Train the model
    trainer.fit(model, datamodule=datamodule)
    
    # Test the model
    if cfg.get('run_test', False):
        trainer.test(model, datamodule=datamodule)
    
    # Return the best checkpoint path if available
    for callback in trainer.callbacks:
        if hasattr(callback, 'best_model_path'):
            print(f"Best model saved at: {callback.best_model_path}")
            return callback.best_model_path
    
    return None


if __name__ == "__main__":
    train()
