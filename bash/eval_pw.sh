python scripts/eval_pw.py --log_name eval_pw_b100.txt   --eval_dir new_trained/final_no/MNIST/1/cnn_small/simple/k_50  --device cuda:0 --beta 100
python scripts/eval_pw.py --log_name eval_pw_b100.txt   --eval_dir new_trained/final_500/MNIST/1/cnn_small/simple/k_50 --device cuda:0 --beta 100

python scripts/eval_pw.py --log_name eval_pw_b100.txt   --eval_dir new_trained/final_no/MNIST/1/cnn_small/simple/k_100  --device cuda:0 --beta 100
python scripts/eval_pw.py --log_name eval_pw_b100.txt   --eval_dir new_trained/final_500/MNIST/1/cnn_small/simple/k_100 --device cuda:0 --beta 100

python scripts/eval_pw.py --log_name eval_pw_b100.txt   --eval_dir new_trained/final_no/CIFAR/1/VGG16/simple/k_50  --device cuda:0 --beta 100
python scripts/eval_pw.py --log_name eval_pw_b100.txt   --eval_dir new_trained/final_500/CIFAR/1/VGG16/simple/k_50 --device cuda:0 --beta 100