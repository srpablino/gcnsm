# Graph Convolutional Networks - Schema Matching (GCNSM)

<p>For the case of the experiment with the OML datasets from <a href="https://github.com/AymanUPC/all_prox_openml/tree/master/OML02">this work</a>, you can skip step1 and step2 by downloading the files indicated in the folder /word_embeddings</p>

<p>PD: For now, you should skip to step3 directly. Step1 and step2 are still incomplete, they were already executed for the OML experiment case in particular </p>

## Step 1:
<p> Extract metafeatures from datasets with pandas profiling tool </p>

## Step 2: 
<p> Build graph from the datasets and metafeatures + encoded features of the datasets using word embeddings (fasttext or bert) </p>

## Step 2.5: 
<p> Based on the graphs you encoded, you will need to provide a dataset indicating the pairs relationship among all the graphs (1 if pair is possitve, 0 otherwise ). You should put this dataset inside the /dataset folder </p>

<p> The name of the columns for this dataset should be: ["dataset1_id", "dataset2_id","matching_topic"] </p>

## Step 3: 
<p> Use the encoded graph to train a NN which will learn how the input datasets are related, and be able to relate new unseen datasets to the ones you already have. This step requieres that you have a dataset of related pairs and the encoded graphs with fasttext and/or bert </p>

