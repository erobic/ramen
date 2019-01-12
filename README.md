## Setup environment variables and data
#### Step 1 - Setup project variables
- Edit `scripts/common.sh` and define all the variables. These variables will be imported by every script we run.

(Note that, I keep source code and results in HDD, data in SSD. Putting these in 3 different locations makes it very easy to copy to other machines whenever we want.)

#### Step 2 - Setup Questions/Annotations Files
- Download the questions/annotations files and put them into ${DATA_ROOT}/questions directory

- Rename the files as: 
    - train_questions.json, val_questions.json, test_questions.json
    - train_annotations.json, val_annotations.json


#### Step 3 - Create a new conda environment and install the requirements
- `pip install  -r requirements.txt`

## Preprocess
#### Extract image features
TODO

#### Extract question features
- `./scripts/preprocess_questions.sh`

## Experiment Naming 
