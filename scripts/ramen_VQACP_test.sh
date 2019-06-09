#!/usr/bin/env bash
set -e
source scripts/common.sh
cd ${PROJECT_ROOT}


DATA_SET=VQACP
DATA_ROOT=${ROOT}/${DATA_SET}

# Test the model
RESULTS_ROOT=${ROOT}/${DATA_SET}_results
MODEL=Ramen
RESUME_EXPT_NAME=${MODEL}_${DATA_SET}_dropout
EXPT_NAME=${RESUME_EXPT_NAME}_test

CUDA_VISIBLE_DEVICES=2 python -u run_network.py \
--data_set ${DATA_SET} \
--data_root ${DATA_ROOT} \
--expt_name ${EXPT_NAME} \
--test \
--resume_expt_name ${RESUME_EXPT_NAME} \
--model ${MODEL} > ${RESULTS_ROOT}/${EXPT_NAME}.log