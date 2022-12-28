#!/bin/bash

# DATASET="proofwriter_cwa_d0"
# DATASET_TYPE="proofwriter"
DATASET="clutrr_4_hop"
DATASET_TYPE="clutrr"
INPUT_FORMAT="lm"
MODEL_TYPE="gpt2"
MODEL_NAME_OR_PATH="gpt2"
TRAIN_BATCH_SIZE=2
PREDICT_BATCH_SIZE=1
INNER_MODE="open"
GD_ACCUMULATE_STPES=1
INNER_STEPS=6
INNER_OPT="adam"
#POSTFIX="no-facts"
#POSTFIX="baseline"
POSTFIX="adam-4-step-t2"
CHEKPOINT="./output/model.ckpt"

echo "Downloading data..."
mkdir -p data/${DATASET}
wandb artifact get causal_scaffold/data_uploader/${DATASET}:latest --root data/${DATASET}

python cli_maml.py \
    --do_train \
    --dataset ${DATASET} \
    --dataset_type ${DATASET_TYPE} \
    --model_type ${MODEL_TYPE} \
    --input_format ${INPUT_FORMAT} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --predict_batch_size ${PREDICT_BATCH_SIZE} \
    --learning_rate 3e-5 \
    --gradient_accumulation_steps ${GD_ACCUMULATE_STPES} \
    --wandb_name ${INNER_MODE}-${DATASET}-${POSTFIX} \
    --inner_mode ${INNER_MODE} \
    --inner_opt ${INNER_OPT} \
    --n_inner_iter ${INNER_STEPS} \
    --callback_monitor val_acc \
    --wandb_checkpoint \
    --device_idx 0
    # --load_checkpoint ${CHEKPOINT}
    # --align
    # --baseline
    # --no_facts \
    # --align \
    # --load_checkpoint ${CHEKPOINT}
    # --freeze_partial
    # --do_eval