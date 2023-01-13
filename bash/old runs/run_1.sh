python scripts/train.py  --cfg_name cnn_small --trunc_type conv     --dataset MNIST --exp newest_adv --k 12 --perturb 12 --lr 0.01 --bs 256 --iters 4 --seed 99 --device cuda:0
python scripts/train.py  --cfg_name cnn_small --trunc_type conv     --dataset MNIST --exp newest_adv --k 12 --perturb 12 --lr 0.01 --bs 256 --iters 4 --seed 1  --device cuda:0
python scripts/train.py  --cfg_name cnn_small --trunc_type conv     --dataset MNIST --exp newest_adv --k 12 --perturb 12 --lr 0.01 --bs 256 --iters 4 --seed 2  --device cuda:0
