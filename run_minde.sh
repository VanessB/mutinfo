cd source/evaluate
export CUDA_VISIBLE_DEVICES=1
python3 run.py --config-name=mixing.yaml +estimator=MINDE-Conv2d +distribution=mixing/label/MNIST +mixing=label/SymmetricNoisyChannel ++estimator.device="cuda:0"