python scripts/eval_rs.py --log_name eval_rs_b1.txt   --eval_dir new_trained/final_no/ --device cuda:0 --beta 1
python scripts/eval_rs.py --log_name eval_rs_b1.txt   --eval_dir new_trained/final_500/ --device cuda:0 --beta 1
python scripts/eval_rs.py --log_name eval_rs_b100.txt --eval_dir new_trained/final_no/ --device cuda:0 --beta 100
python scripts/eval_rs.py --log_name eval_rs_b100.txt --eval_dir new_trained/final_500/ --device cuda:0 --beta 100
python scripts/eval_rs.py --log_name eval_rs_b05.txt  --eval_dir new_trained/final_no/ --device cuda:0 --beta 0.5
python scripts/eval_rs.py --log_name eval_rs_b05.txt  --eval_dir new_trained/final_500/ --device cuda:0 --beta 0.5