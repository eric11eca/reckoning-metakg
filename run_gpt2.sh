DATASET="clutrr_simple"
DATASET_TYPE="clutrr"
INPUT_FORMAT="lm"
MODEL_TYPE="gpt2"
MODEL_NAME_OR_PATH="gpt2"
TRAIN_BATCH_SIZE=4
PREDICT_BATCH_SIZE=1
INNER_MODE="open"
GD_ACCUMULATE_STPES=1
INNER_STEPS=1
POSTFIX="single-head-relation"


python cli_maml.py \
    --do_train \
    --dataset ${DATASET} \
    --dataset_type ${DATASET_TYPE} \
    --model_type ${MODEL_TYPE} \
    --input_format ${INPUT_FORMAT} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --predict_batch_size ${PREDICT_BATCH_SIZE} \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps ${GD_ACCUMULATE_STPES} \
    --wandb_name ${INNER_MODE}-${DATASET}-${POSTFIX} \
    --inner_mode ${INNER_MODE} \
    --n_inner_iter ${INNER_STEPS} \
    --callback_monitor val_f1 \
    --device_idx 0
    # --load_checkpoint output/20221129-195134/epoch=0-step=9000.ckpt
    # --freeze_partial
    # --do_eval