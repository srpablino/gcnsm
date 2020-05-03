# Graph Convolutional Networks - Schema Matching (GCNSM)

For the case of the experiment with the OML datasets from [this work](https://github.com/AymanUPC/all_prox_openml/tree/master/OML02), you can skip step1 and step2 by downloading the files indicated in the folder /word_embeddings

PD: For now, you should skip to step3 directly. Step1 and step2 are still incomplete, they were already executed for the OML experiment case in particular

## Step 1:
Extract metafeatures from datasets with pandas profiling tool

## Step 2: 
Build graph from the datasets and metafeatures + encoded features of the datasets using word embeddings (fasttext or bert)

## Step 2.5: 
Based on the graphs you encoded, you will need to provide a dataset indicating the pairs relationship among all the graphs (1 if pair is possitve, 0 otherwise ). You should put this dataset inside the /dataset folder

The name of the columns for this dataset should be: ["dataset1_id", "dataset2_id","matching_topic"]

## Step 3: 
Use the encoded graph to train a NN which will learn how the input datasets are related, and be able to relate new unseen datasets to the ones you already have. This step requieres that you have a dataset of related pairs and the encoded graphs with fasttext and/or bert (see /word_embeddings/readme.txt)

To run the experiment configure the enviroment with the file: step3_dataset_setup.py. Then run the python notebook: step3_run_tests.ipynb. You can plot the results with the step3_plot_results.ipynb notebook (you need to specify the path where the logs of results are)
