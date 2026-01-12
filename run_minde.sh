cd source/evaluate
export CUDA_VISIBLE_DEVICES=3
python3 run.py --config-name=mixing.yaml\
 +distribution=mixing/label/MNIST\
 +mixing=label/SymmetricNoisyChannel\
 estimator.backbone_factory.hidden_dim=64\
 estimator.backbone_factory.layers_per_block=2\
 +estimator=MINDE-UNet ++n_samples=50000

# python3 run.py --config-name=mixing.yaml +estimator=MINDE-MLP +distribution=mixing/label/MNIST +mixing=label/SymmetricNoisyChannel
