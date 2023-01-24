#!/bin/bash

DATA_PATH=$1
DATA_NAME=$2

if [ "${DATA_PATH}" == "" ]; then
    exit "please specify target directory"
fi
if [ "${DATA_NAME}" == "" ]; then
    exit "please specify dataset name"
fi

export WANDB_ENTITY="epfl_nlp_phd"
export WANDB_PROJECT="data-collection"

wandb artifact put --name ${2} ${1}
