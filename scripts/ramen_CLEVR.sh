#!/usr/bin/env bash

source scripts/common.sh
cd ${PROJECT_ROOT}

DATASET=CLEVR
DATA_ROOT=/hdd/robik/${DATASET}
RESULTS_ROOT=/hdd/robik//${DATASET}_results
mkdir -p ${RESULTS_ROOT}
MODEL=Ramen
python -u run_network.py \
--data_root ${DATA_ROOT} \
--expt_name ${MODEL} \
--model ${MODEL} \
--h5_prefix all > ${RESULTS_ROOT}/${MODEL}.log