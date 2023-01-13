python scripts/train.py --cfg_name cnn_small --dataset MNIST --no_adv --exp final_no --lr 0.01 --bs 256 --epochs 20 --queries 500 --trunc_type simple --k 25 --device cuda:1
python scripts/train.py --cfg_name cnn_small --dataset MNIST --no_adv --exp final_no --lr 0.01 --bs 256 --epochs 20 --queries 500 --trunc_type simple --k 50 --device cuda:1
python scripts/train.py --cfg_name cnn_small --dataset MNIST --no_adv --exp final_no --lr 0.01 --bs 256 --epochs 20 --queries 500 --trunc_type simple --k 75 --device cuda:1

python scripts/train.py --cfg_name VGG16 --dataset CIFAR --no_adv --exp final_no --lr 0.1 --bs 128 --epochs 20 --queries 500 --trunc_type simple --k 25 --device cuda:1
python scripts/train.py --cfg_name VGG16 --dataset CIFAR --no_adv --exp final_no --lr 0.1 --bs 128 --epochs 20 --queries 500 --trunc_type simple --k 50 --device cuda:1
python scripts/train.py --cfg_name VGG16 --dataset CIFAR --no_adv --exp final_no --lr 0.1 --bs 128 --epochs 20 --queries 500 --trunc_type simple --k 75 --device cuda:1



