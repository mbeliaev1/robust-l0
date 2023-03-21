python scripts/eval_rs.py --log_name eval_rs.txt   --eval_dir new_trained/final_long/CIFAR/1/VGG16/og  --device cuda:1 --beta 1
python scripts/eval_rs.py --log_name eval_rs.txt   --eval_dir new_trained/final_long/MNIST/1/cnn_small/og  --device cuda:1 --beta 1


python scripts/eval_pw.py --log_name eval_pw.txt   --eval_dir new_trained/final_long/CIFAR/1/VGG16/og  --device cuda:0
python scripts/eval_pw.py --log_name eval_pw.txt   --eval_dir new_trained/final_long/MNIST/1/cnn_small/og  --device cuda:0 
# RUNNING ABOVE SCRIPTS ALREADY

# NOT RUNNING BELOW SCRIPTS YET!
python scripts/multi_rs.py --eval_dir new_trained/final_long --log_name final_rs_beta1 --beta 1 --device cuda:0
python scripts/multi_rs.py --eval_dir new_trained/final_long --log_name final_rs_beta100 --beta 100 --device cuda:0

# THESE SHOULD BE DONE ASAP
python scripts/eval_rs.py --log_name eval_rs.txt   --eval_dir new_trained/final_long/CIFAR/1/VGG16/simple  --device cuda:1 --beta 1
python scripts/eval_rs.py --log_name eval_rs.txt   --eval_dir new_trained/final_long/MNIST/1/cnn_small/simple  --device cuda:1 --beta 1


python scripts/eval_pw.py --log_name eval_pw.txt   --eval_dir new_trained/final_long/CIFAR/1/VGG16/simple  --device cuda:0
python scripts/eval_pw.py --log_name eval_pw.txt   --eval_dir new_trained/final_long/MNIST/1/cnn_small/simple  --device cuda:0 