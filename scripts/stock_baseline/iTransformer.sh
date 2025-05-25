model_name=iTransformer

root_path_name=./dataset/NASDAQ_NYSE
data_path_name=stock_sup/2013-01-01
model_id_name=NASDAQ
data_name=NASDAQ_baseline
market=NYSE
graph=False
seq_len=120

market=NYSE
pred_len=1
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
    --des 'testflops' \
    --itr 1 \
    --n_heads 16 \
    --d_model 24 \
    --dropout 0.3 \
    --patch_len 8 --stride 8 --VPT_mode 0 --ATSP_solver SA \
    --train_epochs 50 --patience 5 --batch_size 8 --learning_rate 0.0001 \
    --gpu 7 --loss MSE --market $market