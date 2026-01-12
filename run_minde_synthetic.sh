cd source/evaluate
export CUDA_VISIBLE_DEVICES=3
# python3 run.py --config-name=mixing.yaml +estimator=MINDE-Conv2d +distribution=mixing/label/MNIST +mixing=label/SymmetricNoisyChannel
python3 run.py --config-name=lowdim.yaml +estimator=MINDE-MLP +distribution=CorrelatedNormal
