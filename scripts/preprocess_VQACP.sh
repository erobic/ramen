#!/usr/bin/env bash
source scripts/common.sh

python preprocess/create_dictionary.py --data_root ${DATA_ROOT}
python preprocess/compute_softscore.py --data_root ${DATA_ROOT} --top_k 1000
