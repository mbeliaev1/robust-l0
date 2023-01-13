python scripts/multi_rs.py --eval_dir new_trained/final_500 --log_name cont_multi_rs_1 --beta 1 --device cuda:1
python scripts/multi_rs.py --eval_dir new_trained/final_no --log_name cont_multi_rs_1 --beta 1 --device cuda:1

python scripts/multi_rs.py --eval_dir new_trained/final_500 --log_name cont_multi_rs_100 --beta 100 --device cuda:1
python scripts/multi_rs.py --eval_dir new_trained/final_no --log_name cont_multi_rs_100  --beta 100 --device cuda:1