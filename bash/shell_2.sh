python scripts/train.py --no_adv --cfg_name mlp --trunc_type linear_ind --dataset MNIST --exp no_adv --k 0 --perturb 12 --bs 256 --lr 0.001 --device cuda:1
python scripts/train.py --no_adv --cfg_name mlp --trunc_type linear_dep --dataset MNIST --exp no_adv --k 12 --perturb 12 --bs 256 --lr 0.001 --device cuda:1
python scripts/train.py --no_adv --cfg_name mlp --trunc_type linear_ind --dataset MNIST --exp no_adv --k 12 --perturb 12 --bs 256 --lr 0.001 --device cuda:1
python scripts/train.py --no_adv --cfg_name mlp --trunc_type clip --dataset MNIST --exp no_adv --k 12 --perturb 12 --bs 256 --lr 0.001 --device cuda:1
python scripts/train.py --no_adv --cfg_name mlp --trunc_type linear_dep_abs --dataset MNIST --exp no_adv --k 12 --perturb 12 --bs 256 --lr 0.001 --device cuda:1
python scripts/train.py --no_adv --cfg_name mlp --trunc_type clip_abs --dataset MNIST --exp no_adv --k 12 --perturb 12 --bs 256 --lr 0.001 --device cuda:1