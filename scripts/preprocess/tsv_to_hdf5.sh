#!/usr/bin/env bash

source scripts/common.sh
python preprocess/tsv_to_hdf5.py --data_root ${DATA_ROOT} --split trainval
python preprocess/tsv_to_hdf5.py --data_root ${DATA_ROOT} --split test2015