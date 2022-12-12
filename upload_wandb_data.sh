#!/bin/bash

DATA_PATH=$1
DATA_NAME=$2

WANDB_API_KEY="5d22b1d85f1fd5bb0c5758b93903c364ee5dc93d"

if [ "${WANDB_API_KEY}" == "" ]; then
    exit "please specify wandb api key"
fi
if [ "${DATA_PATH}" == "" ]; then
    exit "please specify target directory"
fi
if [ "${DATA_NAME}" == "" ]; then
    exit "please specify dataset name"
fi

export WANDB_ENTITY="causal_scaffold"
export WANDB_PROJECT="data_uploader"

wandb artifact put --name ${2} ${1}
