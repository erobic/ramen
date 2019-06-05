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

#### Step 4 - Extract question features
`./scripts/preprocess/preprocess_questions.sh`

#### Step 5 - Extract image features
##### For VQAv1/VQAv2/C-VQA and VQA-CP
Download train+val features into ```${FEATURES_DIR}``` from the following link:
https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip

##### Create hdf5 files
Execute the following script to extract the zip files and create hdf5 files ```./tsv_to_h5.sh```

Train and val features are extracted to ```${DATA_ROOT}/features/trainval.hdf5```. The script will create softlinks ```train.hdf5``` and ```val.hdf5```, pointing to ```trainval.hdf5```.

Test features are extracted to ```${DATA_ROOT}/features/test.hdf5```

#### Step 6 - Train
Run the following command (check `scripts/demo/updn.sh`):

`python run_network.py --data_root ${DATA_ROOT} --expt_name UpDn --model UpDn --h5_prefix all`


If image features are in separate files per split (e.g., train.hdf5, val.hdf5, test.hdf5)

`python run_network.py --data_root ${DATA_ROOT} --expt_name UpDn --model UpDn --h5_prefix use_split`

##### To resume

- To resume from checkpoint of same experiment

`python run_network.py --data_root ${DATA_ROOT} --expt_name UpDn --model UpDn --h5_prefix all --resume --resume_expt_name UpDn`

- To resume from checkpoint of a different experiment

`python run_network.py --data_root ${DATA_ROOT} --expt_name UpDn --model UpDn --h5_prefix all --resume --resume_expt_name DifferentUpDn`

##### Conventions
To track all of the experiments properly, let us follow this convention/structure:
- For each new "group of experiments" (e.g., trying to find the best optimizer), create a directory inside `scripts` directory
- Create a new script with a unique name e.g. `scripts/optim/adam.sh` and when training, pass in the same name in `--expt_name` parameter (more details below)

  
#### Step 7 - Test
TODO

### CLEVR
#### Extract UpDn features for CLEVR