#!/bin/bash
model_name=FinD3
gpu=0

root_path_name=./dataset/NASDAQ_NYSE
data_path_name=stock_sup/2013-01-01
model_id_name=NYSEtest
data_name=NASDAQ
market=NYSE
target=Close

pred_len=1
kernel_size=15

d_model=24
hidden_dim=48
seq_len=120
seed=2025
lr=0.00015
alpha=9
gum_bais=5

python -u run.py --graph \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --enc_in 5 \
      --dec_in 5 \
      --c_out 5 \
      --model_id $model_id_name \
      --model $model_name \
      --data $data_name \
      --target $target \
      --features MS \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers 1 \
      --d_layers 1 \
      --factor 1 \
      --des "final" \
      --itr 3 \
      --d_model $d_model \
      --patch_len 8 --stride 8 \
      --dropout 0.3 \
      --VPT_mode 0 --ATSP_solver SA \
      --train_epochs 50 --patience 10 --batch_size 1 --learning_rate $lr \
      --gpu $gpu --alpha $alpha --gum_bais $gum_bais \
      --n_heads 4 --hidden_dim 48 --loss rank --market $market --decomp 1 --k 10 \
      --ltedge 2048 --share_emb 1
