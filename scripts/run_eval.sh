#!/bin/bash

DATASET="owa_proof_5_hop_dall"
DATASET_TYPE="proofwriter"
INPUT_FORMAT="lm"
MODEL_TYPE="gpt2"
MODEL_NAME_OR_PATH="gpt2"
TRAIN_BATCH_SIZE=1
PREDICT_BATCH_SIZE=2
INNER_MODE="all"
GD_ACCUMULATE_STPES=1
INNER_STEPS=5
INNER_OPT="adam"
#POSTFIX="no-facts"
#POSTFIX="baseline"
POSTFIX="test"
CHEKPOINT="./output/model.ckpt"
 
echo "Downloading data..."
mkdir -p data/${DATASET}
wandb artifact get epfl_nlp_phd/data-collection/${DATASET}:latest --root data/${DATASET}

echo "Downloading model..."
wandb artifact get epfl_nlp_phd/meta-knowledge/owa_proof_5_hop_dall:best_k --root ./output/

LOAD_ORDER="norm"
python cli_maml.py \
    --do_eval \
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
    --lora_r ${LORA_R} \
    --load_order ${LOAD_ORDER} \
    --dyna_lr \
    --multi_task \
    --load_checkpoint ${CHEKPOINT}

# LOAD_ORDER="pre"
# python cli_maml.py \
#     --do_eval \
#     --dataset ${DATASET} \
#     --dataset_type ${DATASET_TYPE} \
#     --model_type ${MODEL_TYPE} \
#     --input_format ${INPUT_FORMAT} \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --train_batch_size ${TRAIN_BATCH_SIZE} \
#     --predict_batch_size ${PREDICT_BATCH_SIZE} \
#     --learning_rate 3e-5 \
#     --num_train_epochs 1 \
#     --gradient_accumulation_steps ${GD_ACCUMULATE_STPES} \
#     --wandb_name ${POSTFIX}-${LOAD_ORDER} \
#     --inner_mode ${INNER_MODE} \
#     --inner_opt ${INNER_OPT} \
#     --n_inner_iter ${INNER_STEPS} \
#     --callback_monitor val_acc \
#     --device_idx 0 \
#     --multi_task \
#     --load_checkpoint ${CHEKPOINT} \
#     --load_order ${LOAD_ORDER}

# LOAD_ORDER="post"
# python cli_maml.py \
#     --do_eval \
#     --dataset ${DATASET} \
#     --dataset_type ${DATASET_TYPE} \
#     --model_type ${MODEL_TYPE} \
#     --input_format ${INPUT_FORMAT} \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --train_batch_size ${TRAIN_BATCH_SIZE} \
#     --predict_batch_size ${PREDICT_BATCH_SIZE} \
#     --learning_rate 3e-5 \
#     --num_train_epochs 1 \
#     --gradient_accumulation_steps ${GD_ACCUMULATE_STPES} \
#     --wandb_name ${POSTFIX}-${LOAD_ORDER} \
#     --inner_mode ${INNER_MODE} \
#     --inner_opt ${INNER_OPT} \
#     --n_inner_iter ${INNER_STEPS} \
#     --callback_monitor val_acc \
#     --device_idx 0 \
#     --multi_task \
#     --load_checkpoint ${CHEKPOINT} \
#     --load_order ${LOAD_ORDER}