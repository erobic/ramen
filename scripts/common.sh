#!/usr/bin/env bash
source activate vqa2
# I keep source code and results in HDD, data in SSD. Putting these in 3 different locations makes it very easy to copy to other machines whenever we want.
PROJECT_ROOT=/hdd/robik/projects/ramen
DATASET=VQACP_1000
DATA_ROOT=/hdd/robik/${DATASET}
RESULTS_ROOT=/hdd/robik/${DATASET}_results