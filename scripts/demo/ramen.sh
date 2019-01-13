#!/usr/bin/env bash

source scripts/common.sh
cd ${PROJECT_ROOT}

# Use the following if image features are in same file: all.hdf5 (default is all.hdf5)
python run_network.py --data_root ${DATA_ROOT} --expt_name Ramen --model Ramen --h5_prefix all