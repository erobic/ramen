#!/usr/bin/env bash
set -e
source scripts/common.sh
cd ${PROJECT_ROOT}


DATA_SET=CVQA
DATA_ROOT=${ROOT}/${DATA_SET}

# Create dictionary and compute GT answer scores
#python preprocess/create_dictionary.py --data_root ${DATA_ROOT}
#python preprocess/compute_softscore.py --data_root ${DATA_ROOT} --top_k 1000

# Train the model
RESULTS_ROOT=${ROOT}/${DATA_SET}_results
mkdir -p ${RESULTS_ROOT}
MODEL=Ramen
EXPT_NAME=${MODEL}_${DATA_SET}

CUDA_VISIBLE_DEVICES=1 python -u run_network.py \
--data_set ${DATA_SET} \
--data_root ${DATA_ROOT} \
--expt_name ${EXPT_NAME} \
--model ${MODEL} > ${RESULTS_ROOT}/${EXPT_NAME}.log