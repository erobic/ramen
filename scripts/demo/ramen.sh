#!/usr/bin/env bash

source scripts/common.sh
cd ${PROJECT_ROOT}

MODEL=Ramen
python -u run_network.py \
--data_root ${DATA_ROOT} \
--expt_name ${MODEL} \
--model ${MODEL} > ${RESULTS_ROOT}/${MODEL}.log