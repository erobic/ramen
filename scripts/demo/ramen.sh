#!/usr/bin/env bash

source scripts/common.sh
cd ${PROJECT_ROOT}

# Use the following if image features are in same file: all.hdf5 (default is all.hdf5)
MODEL=Ramen
python -u run_network.py \
--data_root ${DATA_ROOT} \
--expt_name ${MODEL} \
--model ${MODEL} \
--h5_prefix all &> ${RESULTS_ROOT}/${MODEL}.log