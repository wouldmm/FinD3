export CUDA_VISIBLE_DEVICES=0

model_name=Autoformer

root_path_name=./dataset/NASDAQ_NYSE
data_path_name=stock_sup/2013-01-01
model_id_name=test
data_name=NASDAQ_baseline
seq_len=512

pred_len=24
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --enc_in 5 \
    --dec_in 5 \
    --c_out 5 \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --e_layers 5 \
    --d_layers 2 \
    --factor 1 \
    --des 'Exp' \
    --itr 1 \
    --n_heads 16 \
    --d_model 16 \
    --dropout 0.3 \
    --patch_len 48 --stride 48 --VPT_mode 0 --ATSP_solver SA \
    --train_epochs 50 --patience 5 --batch_size 32 --learning_rate 0.001 \
    --gpu 0 --loss MSE --graph False