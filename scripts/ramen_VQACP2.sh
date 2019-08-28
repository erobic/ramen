#!/usr/bin/env bash
set -e
source scripts/common.sh
cd ${PROJECT_ROOT}


DATA_SET=VQACP
DATA_ROOT=${ROOT}/${DATA_SET}

# Create dictionary and compute GT answer scores
#python preprocess/create_dictionary.py --data_root ${DATA_ROOT}
#python preprocess/compute_softscore.py --data_root ${DATA_ROOT} --top_k 3000

## Train the model
RESULTS_ROOT=${ROOT}/${DATA_SET}_results
MODEL=Ramen
EXPT_NAME=${MODEL}_${DATA_SET}_bs128
CUDA_VISIBLE_DEVICES=0 python -u run_network.py \
--data_set ${DATA_SET} \
--data_root ${DATA_ROOT} \
--expt_name ${EXPT_NAME} \
--epochs 35 \
--words_dropout 0.5 \
--question_dropout_after_rnn 0.5 \
--batch_size 128 \
--model ${MODEL} > ${RESULTS_ROOT}/${EXPT_NAME}.log

# --words_dropout 0.5 \
##!/usr/bin/env bash
#set -e
#source scripts/common.sh
#cd ${PROJECT_ROOT}
#
#
#DATA_SET=VQACP
#DATA_ROOT=${ROOT}/${DATA_SET}
#
## Create dictionary and compute GT answer scores
#python preprocess/create_dictionary.py --data_root ${DATA_ROOT}
#python preprocess/compute_softscore.py --data_root ${DATA_ROOT} --top_k 1000
#
## Train the model
#RESULTS_ROOT=${ROOT}/${DATA_SET}_results
#MODEL=Ramen
#EXPT_NAME=${MODEL}_${DATA_SET}_cosine
#CUDA_VISIBLE_DEVICES=2 python -u run_network.py \
#--data_set ${DATA_SET} \
#--data_root ${DATA_ROOT} \
#--expt_name ${EXPT_NAME} \
#--disable_batch_norm_for_late_fusion \
#--classifier_type CosineLinear \
#--model ${MODEL} > ${RESULTS_ROOT}/${EXPT_NAME}.log