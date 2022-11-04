python scripts/train.py --no_adv --cfg_name cnn_small --trunc_type conv --exp cnn_no_adv --k 12 --perturb 12 --device cuda:1 --seed 0
python scripts/train.py --no_adv --cfg_name cnn_small --trunc_type conv --exp cnn_no_adv --k 12 --perturb 12 --device cuda:1 --seed 1
python scripts/train.py --no_adv --cfg_name cnn_small --trunc_type clip --exp cnn_no_adv --k 12 --perturb 12 --device cuda:1 --seed 0
python scripts/train.py --no_adv --cfg_name cnn_small --trunc_type clip --exp cnn_no_adv --k 12 --perturb 12 --device cuda:1 --seed 1
python scripts/train.py --no_adv --cfg_name cnn_small --trunc_type conv --exp cnn_no_adv --k 0  --perturb 12 --device cuda:1 --seed 0
python scripts/train.py --no_adv --cfg_name cnn_small --trunc_type conv --exp cnn_no_adv --k 0  --perturb 12 --device cuda:1 --seed 1

