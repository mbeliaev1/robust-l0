python scripts/train.py  --cfg_name VGG16     --trunc_type clip     --dataset CIFAR --exp newest_adv --k 12 --perturb 12 --lr 0.1  --bs 128 --iters 8 --seed 3  --device cuda:1
python scripts/train.py  --cfg_name VGG16     --trunc_type simple   --dataset CIFAR --exp newest_adv --k 12 --perturb 12 --lr 0.1  --bs 128 --iters 8 --seed 3  --device cuda:1

python scripts/train.py  --cfg_name cnn_small --trunc_type simple   --dataset MNIST --exp newest_adv --k 12 --perturb 12 --lr 0.01 --bs 256 --iters 4 --seed 3  --device cuda:1
python scripts/train.py  --cfg_name cnn_small --trunc_type clip     --dataset MNIST --exp newest_adv --k 12 --perturb 12 --lr 0.01 --bs 256 --iters 4 --seed 3  --device cuda:1






























