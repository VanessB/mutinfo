cd source/evaluate
export CUDA_VISIBLE_DEVICES=0

MI_VALUES=(2.3)

for MI in "${MI_VALUES[@]}"; do
    python3 run.py --config-name=mixing.yaml\
    +distribution=mixing/label/MNIST\
    +mixing=label/SymmetricNoisyChannel\
    estimator.backbone_factory.hidden_dim=64\
    estimator.backbone_factory.layers_per_block=2\
    estimator.variant=j\
    +estimator=MINDE-UNet ++n_samples=50000\
    +default_grid.mutual_information=${MI}
done