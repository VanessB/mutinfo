cd source/evaluate
export CUDA_VISIBLE_DEVICES=6
python3 run.py --config-name=mixing.yaml\
 +distribution=mixing/modulation/MNIST\
 +mixing=modulation/Multiplication\
 estimator.backbone_factory.hidden_dim=64\
 estimator.backbone_factory.layers_per_block=2\
 'estimator.ckpt_path="/home/foresti/mutinfo-minde/source/evaluate/checkpoints/MINDE-Conv2d/mi=1.5/20251008_172839_450448/last.ckpt"'\
 +estimator=MINDE-UNet ++n_samples=50000

# python3 run.py --config-name=mixing.yaml +estimator=MINDE-MLP +distribution=mixing/label/MNIST +mixing=label/SymmetricNoisyChannel