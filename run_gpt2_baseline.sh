DATASET="clutrr"
INPUT_FORMAT="lm"
MODEL_TYPE="gpt2"
MODEL_NAME_OR_PATH="gpt2"
TRAIN_BATCH_SIZE=8
PREDICT_BATCH_SIZE=4


python cli_maml.py \
    --do_train \
    --baseline \
    --dataset ${DATASET} \
    --model_type ${MODEL_TYPE} \
    --input_format ${INPUT_FORMAT} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --predict_batch_size ${PREDICT_BATCH_SIZE} \
    --learning_rate 3e-5 \
    --wandb_name baseline-${DATASET} \
    --device_idx 3
    # --do_eval \
    # --no_facts
    # --classifier \