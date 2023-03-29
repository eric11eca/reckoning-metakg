#!/bin/bash

DATASET="owa_proof_2_hop"
DATASET_TYPE="proofwriter"
INPUT_FORMAT="lm"
MODEL_TYPE="gpt2"
MODEL_NAME_OR_PATH="gpt2-large"
TRAIN_BATCH_SIZE=4
PREDICT_BATCH_SIZE=1
INNER_MODE="lora"
GD_ACCUMULATE_STPES=1
INNER_STEPS=4
INNER_OPT="adam"
#POSTFIX="no-facts"
#POSTFIX="baseline"
PREFIX_DIM=128
LOAD_ORDER="norm"
POSTFIX="multi-lora"
CHEKPOINT="./output/model.ckpt"
 
echo "Downloading data..."
mkdir -p data/${DATASET}
wandb artifact get epfl_nlp_phd/data-collection/${DATASET}:latest --root data/${DATASET}

echo "Downloading model..."
wandb artifact get epfl_nlp_phd/meta-knowledge/proof_2_hop_lora:best_k --root ./output/

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
    --num_train_epochs 10 \
    --gradient_accumulation_steps ${GD_ACCUMULATE_STPES} \
    --wandb_name ${DATASET}-${POSTFIX} \
    --inner_mode ${INNER_MODE} \
    --inner_opt ${INNER_OPT} \
    --n_inner_iter ${INNER_STEPS} \
    --callback_monitor val_acc_label \
    --wandb_checkpoint \
    --device_idx 0 \
    --prefix_dim ${PREFIX_DIM} \
    --multi_task \
    --load_order ${LOAD_ORDER} \
    --load_checkpoint ${CHEKPOINT} \
    # --max_data 2000 \
    # --freeze_partial