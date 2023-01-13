python scripts/eval_pw.py --log_name eval_pw_b1.txt   --eval_dir new_trained/final_no/  --device cuda:1 --beta 1
python scripts/eval_pw.py --log_name eval_pw_b1.txt   --eval_dir new_trained/final_500/ --device cuda:1 --beta 1

python scripts/eval_rs.py --log_name eval_rs_b1.txt   --eval_dir new_trained/final_no/MNIST/1/cnn_small/simple/k_75 --device cuda:1 --beta 1
python scripts/eval_rs.py --log_name eval_rs_b1.txt   --eval_dir new_trained/final_500/MNIST/1/cnn_small/simple/k_75 --device cuda:1 --beta 1