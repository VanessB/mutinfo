cd source/evaluate
export CUDA_VISIBLE_DEVICES=4
python3 run.py --config-name=mixing.yaml +estimator=InfoSEDD-MLP +distribution=mixing/label/MNIST +mixing=label/SymmetricNoisyChannel
# python3 run.py --config-name=mixing.yaml +estimator=MINDE-MLP +distribution=mixing/label/MNIST +mixing=label/SymmetricNoisyChannel
