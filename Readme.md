# Graph Convolutional Networks - Schema Matching (GCNSM)

For the dataset similarity experiments (based on meta-features extracted from OML datasets, see [this work](https://github.com/AymanUPC/all_prox_openml/tree/master/OML02)), you can skip step1 and step2 by downloading the files indicated in the folder step2/output/

## Step 1:
Extract metafeatures from datasets with pandas profiling tool
You need to create a folder (/step1/input/folder_experiment_name/) with the datasets of your experiment inside

## Step 2: 
Build graph from the datasets and metafeatures + encoded features of the datasets using the fastText word embeddings model

## Step 3: 
Use the encoded graph to train a NN which will learn how the input datasets are related, and be able to relate new unseen datasets to the ones you already have. This step requieres that you have the graph with the fasttext-encoded features of your input datasets (see step2/output/readme.txt)

This step also requeres that you have a csv ground_truth (with a header) following the format: \
id1,id2,match \
xx,yy,1 \
xx,zz,-1 

In this simple example, nodes xx and yy are similar (1), while nodes xx and zz are not similar (-1)

### There 3 different cases according to the experiment performed:
#### hold_out experiment:
You will need to create 2 separated files (train.csv, and test.csv) and put them inside step3/ground_truth/folder_experiment_name/hold_out
#### cv_10 and random_subsampling
For the 10-fold cross validation and random subsampling experiments, you will need just one file inside step3/ground_truth/folder_experiment_name/experiment_name.csv (or .json etc), which later will be processed to get the train/test splits, according the specific experiment (10-fold cv or random_sub_sampling). 
Here you need a 4th column, with the topic that would group the nodes that are related:\
id1,id2,match,topic\
xx,yy,1,topic1 \
xx,zz,-1,Null

## Running the experiment
You can run the experiment step by step, using the corresponding jupyter notebooks inside each step folder
