export CUDA_VISIBLE_DEVICES=6

model_name=Autoformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/methane_concentration/FirstVisit \
  --data_path merged_0.csv \
  --model_id traffic_40_40 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len 40 \
  --label_len 20 \
  --pred_len 40 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 3