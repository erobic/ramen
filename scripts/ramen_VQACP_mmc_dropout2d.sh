#!/usr/bin/env bash

source scripts/common.sh
cd ${PROJECT_ROOT}


DATA_SET=VQACP_1000
DATA_ROOT=/hdd/robik/${DATA_SET}

# Create dictionary and compute GT answer scores
#python preprocess/create_dictionary.py --data_root ${DATA_ROOT}
#python preprocess/compute_softscore.py --data_root ${DATA_ROOT} --top_k 1000

# Run RAMEN
RESULTS_ROOT=/hdd/robik/${DATA_SET}_results
MODEL=Ramen
EXPT_NAME=${MODEL}_${DATASET}_mmc_dropout2d_0.5
python -u run_network.py \
--data_set ${DATA_SET} \
--data_root ${DATA_ROOT} \
--expt_name ${EXPT_NAME} \
--mmc_dropout 0.5 \
--h5_prefix all \
--model ${MODEL} > ${RESULTS_ROOT}/${EXPT_NAME}.log