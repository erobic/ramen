#!/usr/bin/env bash

source scripts/common.sh
cd ${PROJECT_ROOT}

DATASET=VQACP_1000
DATA_ROOT=/hdd/robik/${DATASET}

# Create dictionary and compute GT answer scores
#python preprocess/create_dictionary.py --data_root ${DATA_ROOT}
#python preprocess/compute_softscore.py --data_root ${DATA_ROOT} --top_k 1000

# Run RAMEN
RESULTS_ROOT=/hdd/robik/${DATASET}_results
mkdir -p ${RESULTS_ROOT}
MODEL=Ramen
EXPT_NAME=${MODEL}_${DATASET}_all_LSTM
python -u run_network.py \
--data_root ${DATA_ROOT} \
--expt_name ${EXPT_NAME} \
--question_rnn_type LSTM \
--h5_prefix all \
--model ${MODEL} > ${RESULTS_ROOT}/${EXPT_NAME}.log