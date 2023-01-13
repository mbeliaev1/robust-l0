python scripts/train.py --cfg_name cnn_small --trunc_type og         --exp beta_1               --k 0  --beta 1   --device cuda:0 --seed 0
python scripts/train.py --cfg_name cnn_small --trunc_type og         --exp beta_100             --k 0  --beta 100 --device cuda:0 --seed 0
python scripts/train.py --cfg_name cnn_small --trunc_type og         --exp beta_1_no   --no_adv --k 0  --beta 1   --device cuda:0 --seed 0
python scripts/train.py --cfg_name cnn_small --trunc_type og         --exp beta_100_no --no_adv --k 0  --beta 100 --device cuda:0 --seed 0

python scripts/train.py --cfg_name mlp       --trunc_type linear_ind --exp beta_1               --k 12 --beta 1   --device cuda:0 --seed 0
python scripts/train.py --cfg_name mlp       --trunc_type linear_ind --exp beta_100             --k 12 --beta 100 --device cuda:0 --seed 0
python scripts/train.py --cfg_name mlp       --trunc_type linear_ind --exp beta_1_no   --no_adv --k 12 --beta 1   --device cuda:0 --seed 0
python scripts/train.py --cfg_name mlp       --trunc_type linear_ind --exp beta_100_no --no_adv --k 12 --beta 100 --device cuda:0 --seed 0

python scripts/train.py --cfg_name cnn_small --trunc_type clip       --exp beta_1               --k 12 --beta 1   --device cuda:1 --seed 0
python scripts/train.py --cfg_name cnn_small --trunc_type clip       --exp beta_100             --k 12 --beta 100 --device cuda:1 --seed 0
python scripts/train.py --cfg_name cnn_small --trunc_type clip       --exp beta_1_no   --no_adv --k 12 --beta 1   --device cuda:1 --seed 0
python scripts/train.py --cfg_name cnn_small --trunc_type clip       --exp beta_100_no --no_adv --k 12 --beta 100 --device cuda:1 --seed 0

python scripts/train.py --cfg_name cnn_small --trunc_type simple     --exp beta_1               --k 12 --beta 1   --device cuda:1 --seed 0
python scripts/train.py --cfg_name cnn_small --trunc_type simple     --exp beta_100             --k 12 --beta 100 --device cuda:1 --seed 0
python scripts/train.py --cfg_name cnn_small --trunc_type simple     --exp beta_1_no   --no_adv --k 12 --beta 1   --device cuda:1 --seed 0
python scripts/train.py --cfg_name cnn_small --trunc_type simple     --exp beta_100_no --no_adv --k 12 --beta 100 --device cuda:1 --seed 0


