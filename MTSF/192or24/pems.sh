#!/bin/bash
mkdir -p ./logs/PEMS07

export CUDA_VISIBLE_DEVICES=7
model_name=CoIn
seq_lens=(96)
bss=(32)
lrs=(1e-3)
log_dir="./logs/PEMS07/"
layers=(5)
pred_lens=(24)
dropouts=(0.2)
d_models=(512)


for bs in "${bss[@]}"; do
    for lr in "${lrs[@]}"; do
        for layer in "${layers[@]}"; do
            for dropout in "${dropouts[@]}"; do
                for d_model in "${d_models[@]}"; do
                    for pred_len in "${pred_lens[@]}"; do
                        for seq_len in "${seq_lens[@]}"; do
                            python -u run.py \
                            --task_name LTSF \
                            --is_training 1 \
                            --root_path /data/gqyu/dataset/PEMS/ \
                            --data_path PEMS07.npz \
                            --model_id "PEMS07_${seq_len}_${pred_len}" \
                            --model $model_name \
                            --data PEMS  \
                            --features M \
                            --seq_len $seq_len \
                            --pred_len $pred_len \
                            --batch_size $bs \
                            --learning_rate $lr \
                            --layers $layer\
                            --dropout $dropout\
                            --d_model $d_model\
                            --enc_in 883  \
                            --dec_in 883  \
                            --c_out 883  \
                            --des 'Exp' \
                            --train_epochs 30\
                            --use_norm 0\
                            --lradj type2\
                            --itr 1 >"${log_dir}bs${bs}_lr${lr}_lay${layer}_dp${dropout}_dm${d_model}_${pred_len}_${seq_len}.log"
                        done
                    done
                done
            done
        done
    done
done

