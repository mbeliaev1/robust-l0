python scripts/train.py --no_adv --cfg_name VGG16 --dataset CIFAR --exp clean_CNN_2 --perturb 0 --lr 0.1  --bs 128 --device cuda:0
python scripts/train.py --no_adv --cfg_name VGG16 --dataset CIFAR --exp clean_CNN_2 --perturb 0 --lr 0.1  --bs 256 --device cuda:0

python scripts/train.py --no_adv --cfg_name VGG16 --dataset CIFAR --exp clean_CNN_2 --perturb 0 --lr 0.01 --bs 128 --device cuda:1
python scripts/train.py --no_adv --cfg_name VGG16 --dataset CIFAR --exp clean_CNN_2 --perturb 0 --lr 0.01 --bs 256 --device cuda:1

