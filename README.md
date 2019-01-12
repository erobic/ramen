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
- `./scripts/preprocess/preprocess_questions.sh`

#### Training and Testing
To track all of the experiments properly, let us follow this convention/structure:
1. For each new "group of experiments" (e.g., trying to find the best optimizer), create a directory inside `scripts` directory
2. Create a new script with a unique name e.g. `scripts/optim/adam.sh` and when training, pass in the same name in `--expt_name` parameter (more details below)  
