#!/usr/bin/env bash

source scripts/common.sh
cd ${PROJECT_ROOT}

DATASET=CLEVR
DATA_ROOT=/hdd/robik/${DATASET}

# Convert to VQA2-like format
#python -u preprocess/convert_from_clevr_to_vqa_format.py --data_root ${DATA_ROOT}
# Create dictionary and compute GT answer scores
#python preprocess/create_dictionary.py --data_root ${DATA_ROOT}
#python preprocess/compute_softscore.py --data_root ${DATA_ROOT}

RESULTS_ROOT=/hdd/robik/${DATASET}_results

mkdir -p ${RESULTS_ROOT}
MODEL=Ramen
EXPT_NAME=${MODEL}_${DATASET}

python -u run_network.py \
--data_root ${DATA_ROOT} \
--expt_name ${MODEL} \
--model ${MODEL} \
--spatial_feature_type mesh \
--spatial_feature_length 16 \
--h5_prefix use_split > ${RESULTS_ROOT}/${EXPT_NAME}.log