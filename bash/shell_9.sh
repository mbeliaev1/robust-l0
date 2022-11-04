python scripts/train.py --cfg_name cnn_small --trunc_type conv --exp cnn_adv --k 12 --perturb 12 --device cuda:1 --seed 0
python scripts/train.py --cfg_name cnn_small --trunc_type conv --exp cnn_adv --k 12 --perturb 12 --device cuda:1 --seed 1
python scripts/train.py --cfg_name cnn_small --trunc_type clip --exp cnn_adv --k 12 --perturb 12 --device cuda:1 --seed 0
python scripts/train.py --cfg_name cnn_small --trunc_type clip --exp cnn_adv --k 12 --perturb 12 --device cuda:1 --seed 1
python scripts/train.py --cfg_name cnn_small --trunc_type conv --exp cnn_adv --k 0  --perturb 12 --device cuda:1 --seed 0
python scripts/train.py --cfg_name cnn_small --trunc_type conv --exp cnn_adv --k 0  --perturb 12 --device cuda:1 --seed 1

