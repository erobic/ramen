#!/usr/bin/env bash

source scripts/common.sh
cd ${PROJECT_ROOT}

MODEL=UpDn
python -u run_network.py \
--data_root ${DATA_ROOT} \
--expt_name ${MODEL} \
--model ${MODEL} \
--h5_prefix all &> ${RESULTS_ROOT}/${MODEL}.log

# Use the following if image features are in same file: all.hdf5 (default is all.hdf5)
# python run_network.py --data_root ${DATA_ROOT} --expt_name UpDn --model UpDn --h5_prefix all

# Use the following if image features are in separate files per split (e.g., train.hdf5, val.hdf5, test.hdf5)
# python run_network.py --data_root ${DATA_ROOT} --expt_name UpDn --model UpDn --h5_prefix use_split

# To resume
#python run_network.py --data_root ${DATA_ROOT} --expt_name UpDn --model UpDn --h5_prefix all --resume --resume_expt_name UpDn

# To resume from a different checkpoint (best model)
#python run_network.py --data_root ${DATA_ROOT} --expt_name UpDn --model UpDn --h5_prefix all --resume --resume_expt_name DifferentUpDn --resume_expt_type best
