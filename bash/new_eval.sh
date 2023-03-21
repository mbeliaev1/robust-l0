python scripts/eval_rs.py --eval_dir new_trained/final_done --log_name eval_rs.txt --budget 12 --beta 1 --device cuda:1 

python scripts/eval_pw.py --eval_dir new_trained/final_done --log_name eval_pw.txt --device cuda:1

python scripts/multi_rs.py --eval_dir new_trained/final_done --log_name multi_rs_b1.txt --beta 1 --device cuda:0 

python scripts/multi_rs.py --eval_dir new_trained/final_done --log_name multi_rs_b100.txt --beta 100 --device cuda:0 