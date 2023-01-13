python scripts/train.py --cfg_name mlp --trunc_type linear_ind --dataset MNIST --exp adv --k 0 --perturb 12 --bs 256 --lr 0.001 --device cuda:1
python scripts/train.py --cfg_name mlp --trunc_type linear_dep --dataset MNIST --exp adv --k 12 --perturb 12 --bs 256 --lr 0.001 --device cuda:1
python scripts/train.py --cfg_name mlp --trunc_type linear_ind --dataset MNIST --exp adv --k 12 --perturb 12 --bs 256 --lr 0.001 --device cuda:1
python scripts/train.py --cfg_name mlp --trunc_type clip --dataset MNIST --exp adv --k 12 --perturb 12 --bs 256 --lr 0.001 --device cuda:1
python scripts/train.py --cfg_name mlp --trunc_type linear_dep_abs --dataset MNIST --exp adv --k 12 --perturb 12 --bs 256 --lr 0.001 --device cuda:1
python scripts/train.py --cfg_name mlp --trunc_type clip_abs --dataset MNIST --exp adv --k 12 --perturb 12 --bs 256 --lr 0.001 --device cuda:1