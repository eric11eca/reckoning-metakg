#!/bin/bash

DATASET="owa_proof_5_hop"
WANDB_PROJECT="data-collection"
WANDB_ENTITY="epfl_nlp_phd"
 
echo "Downloading data..."
mkdir -p data/${DATASET}
wandb artifact get ${WANDB_ENTITY}/${WANDB_PROJECT}/${DATASET}:latest --root data/${DATASET}

DATASET="owa_proof_5_hop_dall"
mkdir -p data/${DATASET}
wandb artifact get ${WANDB_ENTITY}/${WANDB_PROJECT}/${DATASET}:latest --root data/${DATASET}

DATASET="owa_proof_3_hop"
mkdir -p data/${DATASET}
wandb artifact get ${WANDB_ENTITY}/${WANDB_PROJECT}/${DATASET}:latest --root data/${DATASET}

DATASET="owa_proof_3_hop_dall"
mkdir -p data/${DATASET}
wandb artifact get ${WANDB_ENTITY}/${WANDB_PROJECT}/${DATASET}:latest --root data/${DATASET}

DATASET="owa_proof_2_hop"
mkdir -p data/${DATASET}
wandb artifact get ${WANDB_ENTITY}/${WANDB_PROJECT}/${DATASET}:latest --root data/${DATASET}

DATASET="owa_proof_2_hop_dall"
mkdir -p data/${DATASET}
wandb artifact get ${WANDB_ENTITY}/${WANDB_PROJECT}/${DATASET}:latest --root data/${DATASET}


# echo "Downloading model..."
# wandb artifact get epfl_nlp_phd/meta-knowledge/owa_proof_5_hop_dall:best_k --root ./output/