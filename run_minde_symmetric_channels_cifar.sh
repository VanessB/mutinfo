cd source/evaluate
export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1

MI_VALUES=(0.0)

for MI in "${MI_VALUES[@]}"; do
    python3 run.py --config-name=mixing.yaml\
    +distribution=mixing/label/CIFAR10\
    +mixing=label/SymmetricNoisyChannel\
    estimator.backbone_factory.hidden_dim=64\
    estimator.backbone_factory.layers_per_block=2\
    estimator.variant=c\
    +estimator=MINDE-UNet\
    ++default_grid.mutual_information=${MI}
done