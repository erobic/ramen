#!/usr/bin/env bash

source scripts/common.sh
cd ${PROJECT_ROOT}

# Create dictionary and compute GT answer scores
#python preprocess/create_dictionary.py --data_root ${DATA_ROOT}
#python preprocess/compute_softscore.py --data_root ${DATA_ROOT} --top_k 1000

# Run RAMEN
MODEL=Ramen
EXPT_NAME=${MODEL}_${DATASET}_wd1e-4
python -u run_network.py \
--data_root ${DATA_ROOT} \
--expt_name ${EXPT_NAME} \
--model ${MODEL} > ${RESULTS_ROOT}/${EXPT_NAME}.log