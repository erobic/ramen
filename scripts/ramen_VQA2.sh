#!/usr/bin/env bash

source scripts/common.sh
cd ${PROJECT_ROOT}

DATASET=VQA2
DATA_ROOT=/hdd/robik/${DATASET}

# Create dictionary and compute GT answer scores
#python preprocess/create_dictionary.py --data_root ${DATA_ROOT}
#python preprocess/compute_softscore.py --data_root ${DATA_ROOT} --min_occurrence 9

# Train the model
RESULTS_ROOT=/hdd/robik/${DATASET}_results
mkdir -p ${RESULTS_ROOT}
MODEL=Ramen
EXPT_NAME=${MODEL}_${DATASET}

python -u run_network.py \
--data_root ${DATA_ROOT} \
--expt_name ${MODEL} \
--model ${MODEL} \
--train_split trainval \
--test_split test_dev \
--h5_prefix use_split > ${RESULTS_ROOT}/${EXPT_NAME}.log