#!/bin/bash
mkdir -p ./logs/electricity

export CUDA_VISIBLE_DEVICES=7
model_name=CoIn
seq_lens=(96)
bss=(128)
lrs=(5e-4)
log_dir="./logs/electricity/"
layers=(5)
pred_lens=(96)
dropouts=(0.)
d_models=(512)

for bs in "${bss[@]}"; do
    for lr in "${lrs[@]}"; do
        for layer in "${layers[@]}"; do
            for dropout in "${dropouts[@]}"; do
                for d_model in "${d_models[@]}"; do
                    for pred_len in "${pred_lens[@]}"; do
                        for seq_len in "${seq_lens[@]}"; do
                                python -u run.py \
                                --task_name long_term_forecast \
                                --is_training 1 \
                                --root_path /data/gqyu/dataset/electricity/ \
                                --data_path electricity.csv \
                                --model_id "electricity_${seq_len}_${pred_len}" \
                                --model $model_name \
                                --data custom \
                                --features M \
                                --seq_len $seq_len \
                                --pred_len $pred_len \
                                --batch_size $bs \
                                --learning_rate $lr \
                                --layers $layer\
                                --dropout $dropout\
                                --d_model $d_model\
                                --enc_in 321 \
                                --dec_in 321 \
                                --c_out 321 \
                                --train_epochs 30\
                                --lradj type2\
                                --des 'Exp' \
                                --itr 1 >"${log_dir}bs${bs}_lr${lr}_lay${layer}_dp${dropout}_dm${d_model}_${pred_len}_${seq_len}.log"
                        done
                    done
                done
            done
        done
    done
done
