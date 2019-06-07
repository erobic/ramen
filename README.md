# TODO:
- Uncomment preprocessing
- Add scripts to just test stuff

Let ```ROOT=/hdd/robik/```
### Setting up visual features for VQAv2/CVQA/VQACP
Let ```DATA_ROOT=${ROOT}/${DATASET}```
1. Download train+val features into ```${DATA_DIR}``` [using this link](https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip)
so that you have this file: ```${DATA_DIR}/trainval_36.zip```
2. Extract the zip file, so that you have ```${DATA_DIR}/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv```
3. Download test features into ```${DATA_DIR}``` [using this link](https://imagecaption.blob.core.windows.net/imagecaption/test2015_36.zip)
4. Extract the zip file, so that you have ```${DATA_DIR}/test2015_36/test2015_resnet101_faster_rcnn_genome_36.tsv```
5. Execute the following script to extract the zip files and create hdf5 files ```./tsv_to_h5.sh``` 
    - Train and val features are extracted to ```${DATA_ROOT}/features/trainval.hdf5```. The script will create softlinks ```train.hdf5``` and ```val.hdf5```, pointing to ```trainval.hdf5```.
    - Test features are extracted to ```${DATA_ROOT}/features/test.hdf5```
6. Create the symbolic links/shortcuts:
    - ```${DATA_ROOT}/features/test_dev.hdf5``` (a shortcut to ```${DATA_ROOT}/features/test.hdf5```)
    - ```${DATA_ROOT}/features/test_dev_ids_map.json``` (a shortcut to ```${DATA_ROOT}/features/test_ids_map.json```)

##### VQAv2
1. Download questions and annotations from [this link](https://visualqa.org/download.html).
2. Rename question and annotation files to ```${SPLIT}_questions.json``` and ```${SPLIT}_annotations.json``` and copy them to $ROOT/VQA2/questions. You should have the following files:
    - ```$ROOT/VQA2/questions/train_questions.json```
    - ```$ROOT/VQA2/questions/train_annotations.json```
    - ```$ROOT/VQA2/questions/val_questions.json```
    - ```$ROOT/VQA2/questions/val_annotations.json```
    - ```$ROOT/VQA2/questions/test_questions.json```
    - ```$ROOT/VQA2/questions/test_dev_questions.json```
3. Download [glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip), extract it and copy ```glove.6B.300d.txt``` to ```${DATA_ROOT}/glove/```
4. Execute ```./ramen_VQA2.sh```. This will first preprocess all questions+annotations files and then start training the model.


##### CVQA
1. Download questions and annotations from [this link](https://computing.ece.vt.edu/~aish/cvqa/).
2. Rename:
    - ```cvqa_test_questions.json``` to ```val_questions.json``` 
    - ```cvqa_test_annotations.json``` to ```val_annotations.json```  
3. Rename:
    - ```cvqa_train_questions.json``` to ```train_questions.json``` 
    - ```cvqa_train_annotations.json``` to ```train_annotations.json``` 
4. Copy all of the questions+annotations files to $ROOT/CVQA/questions. You should have the following files:
    - ```$ROOT/CVQA/questions/train_questions.json```
    - ```$ROOT/CVQA/questions/train_annotations.json```
    - ```$ROOT/CVQA/questions/val_questions.json```
    - ```$ROOT/CVQA/questions/val_annotations.json```
5. Download [glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip), extract it and copy ```glove.6B.300d.txt``` to ```${DATA_ROOT}/glove/```
6. Execute ```./ramen_CVQA.sh```. This will first preprocess all questions+annotations files and then start training the model.


### CLEVR
##### Extract bottom-up features for CLEVR
We have provided a [pre-trained FasterRCNN model](https://github.com/erobic/faster_rcnn_1_11_34999/raw/master/faster_rcnn_1_11_34999.pth) and feature extraction script in a [separate repository](https://github.com/erobic/faster-rcnn.pytorch) to extract bottom-up features for CLEVR. 

Please refer to the README file of that repository for detailed instructions to extract the features.

##### Preprocess and Train on CLEVR
1. Download the question files from [this link](https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip)
2. Copy all of the question files to ```${ROOT}/CLEVR/questions```. You should now have the following files:
    - ```$ROOT/CLEVR/questions/CLEVR_train_questions.json``` 
    - ```$ROOT/CLEVR/questions/CLEVR_val_questions.json``` 
    - ```$ROOT/CLEVR/questions/CLEVR_test_questions.json```

3. Download [glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip), extract it and copy ```glove.6B.300d.txt``` to ```${ROOT}/CLEVR/glove/```

4. Before starting the training, make sure you have these files:
    
    - ${ROOT}/CLEVR/ 

##### Execute ```./scripts/ramen_CLEVR.sh``` 

1. This script first converts the CLEVR files into VQA2-like format.
2. Then it creates dictionaries for the dataset.
3. Finally it starts the training. 



## Common
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