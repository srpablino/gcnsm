import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def read_dataset(path,drop_columns=None,keep_columns=None):
    #get rid of useless columns
    csv_data = pd.read_csv(path)
    
    if keep_columns != None:
        #keep only these columns
        return csv_data.filter(items=keep_columns)
    
    if drop_columns!= None:
        #drop these and keep the rest
        return csv_data.drop(drop_columns, axis=1)
    
    #finally, didn't drop or filter any column
    return csv_data

def concat_shuffle(a1,a2):
    out = np.concatenate((a1,a2)) 
    np.random.shuffle(out)
    return out

def sample_negative(triple_list,n=2):
    np.random.shuffle(triple_list)
    t_pos = np.array([x for x in triple_list if x[2]==1])
    t_neg = np.array([x for x in triple_list if x[2]==0])
    n = min(n,int(len(t_neg)/len(t_pos)))
    t_neg = t_neg[0:len(t_pos)*n]
    result = np.concatenate((t_pos,t_neg))
    np.random.shuffle(result)
    return result

def data_augmentation(triple_list):
    t_pos = np.array([x for x in triple_list if x[2]==1])
    t_neg = np.array([x for x in triple_list if x[2]==0])
    result = np.concatenate((t_pos,t_neg))
    ratio = int(len(t_neg) / len(t_pos))
    dif = len(t_neg) - len(t_pos) * ratio
    for i in range (ratio-1):
        np.random.shuffle(t_pos)
        result = np.concatenate((result,t_pos))
    np.random.shuffle(t_pos)    
    result = np.concatenate((result,t_pos[0:dif]))
    np.random.shuffle(result)    
    return result 
 
import os    
from pathlib import Path
def write_files(path,train,test):
    #write files
    df_train = pd.DataFrame(data=train,columns=["dataset1_id","dataset2_id","matching_topic"]) 
    df_test = pd.DataFrame(data=test,columns=["dataset1_id","dataset2_id","matching_topic"]) 
    outdir = "./datasets/"+path
    if not os.path.exists(outdir):
        Path(outdir).mkdir(parents=True, exist_ok=True)
    df_train.to_csv(outdir+"/train.csv",index=False)
    df_test.to_csv(outdir+"/test.csv",index=False)

    
def split_isolation(file_name,neg_sample):
    df_ds = read_dataset("./datasets/"+file_name+".csv",keep_columns=["dataset1_id", "dataset2_id","matching_topic","topic"]).to_numpy()
    df_matching = np.array([x for x in df_ds if x[2] == 1])
    np.random.shuffle(df_matching)
    
    #For each topic with possitive pairs, separate the 20% of node_ids in test
    topics = np.unique(df_ds[:,3])
    topic_pos_test = []
    for t in topics:
        topic_pos_pairs = np.array([x for x in df_matching if x[3] == t])
        if len(topic_pos_pairs) == 0:
            continue
        topic_pos_ds = np.unique(np.concatenate((topic_pos_pairs[:,0],topic_pos_pairs[:,1])))
        if len(topic_pos_ds) > 4:
            np.random.shuffle(topic_pos_ds)
            topic_pos_test.append(topic_pos_ds[0])
    
    #load pairs having the previous separated node-ids in test, the rest in train  
    test = []
    train = []
    for pair in df_ds:
        if pair[0] in topic_pos_test or pair[1] in topic_pos_test:
            test.append(pair)
        else:
            train.append(pair)
    test = np.array(test)
    train = np.array(train)
    
    #sampling neg/pos ratio + data augmentation
    if neg_sample > 0:
        train = sample_negative(train,neg_sample)
    test = sample_negative(test,1)
    train = data_augmentation(train)
    
    path = file_name+"/isolation/"+str(neg_sample)
    write_files(path,train[:,:-1],test[:,:-1])
    print("Train/Test split done")
    return path

def split_cv_isolation(file_name,neg_sample):
    path = file_name+"/isolation/"+str(neg_sample)+"/cv"
    for i in range(10):
        df_ds = read_dataset("./datasets/"+file_name+".csv",keep_columns=["dataset1_id", "dataset2_id","matching_topic","topic"]).to_numpy()
        df_matching = np.array([x for x in df_ds if x[2] == 1])
        np.random.shuffle(df_matching)

        #For each topic with possitive pairs, separate the 20% of node_ids in test
        topics = np.unique(df_ds[:,3])
        topic_pos_test = []
        for t in topics:
            topic_pos_pairs = np.array([x for x in df_matching if x[3] == t])
            if len(topic_pos_pairs) == 0:
                continue
            topic_pos_ds = np.unique(np.concatenate((topic_pos_pairs[:,0],topic_pos_pairs[:,1])))
            if len(topic_pos_ds) > 4:
                np.random.shuffle(topic_pos_ds)
                topic_pos_test.append(topic_pos_ds[0])

        #load pairs having the previous separated node-ids in test, the rest in train  
        test = []
        train = []
        for pair in df_ds:
            if pair[0] in topic_pos_test or pair[1] in topic_pos_test:
                test.append(pair)
            else:
                train.append(pair)
        test = np.array(test)
        train = np.array(train)

        #sampling neg/pos ratio + data augmentation
        if neg_sample > 0:
            train = sample_negative(train,neg_sample)
        test = sample_negative(test,1)
        train = data_augmentation(train)

        write_files(path+"/"+str(i),train[:,:-1],test[:,:-1])
    print("Train/Test split done")
    return path

def split_cv_random(file_name,neg_sample):
    df_ds = read_dataset("./datasets/"+file_name+".csv",keep_columns=["dataset1_id", "dataset2_id","matching_topic"]).to_numpy();
    
    if neg_sample > 0:
        dataset = sample_negative(df_ds,neg_sample)
    
    df_not_matching = np.array([x for x in dataset if x[2]==0])
    df_matching = np.array([x for x in dataset if x[2]==1])

    cv_pos = np.array(np.array_split(df_matching,10))
    cv_neg = np.array(np.array_split(df_not_matching,10))
    path = file_name+"/random/"+str(neg_sample)+"/cv"
    for i in range(10):
        test = np.concatenate((cv_pos[i],cv_neg[i]))
        
        train_pos = np.concatenate((cv_pos[0:i],cv_pos[i+1:]))
        train_neg = np.concatenate((cv_neg[0:i],cv_neg[i+1:]))
        
        train_pos = np.concatenate((train_pos))
        train_neg = np.concatenate((train_neg))
        
        train = np.concatenate((train_pos,train_neg)).squeeze()
        
        test = sample_negative(test,1)
        train = data_augmentation(train)
        
        write_files(path+"/"+str(i),train,test)
            
    
    print("CV Train/Test split done")
    return path
def split_random(file_name,neg_sample):
    df_ds = read_dataset("./datasets/"+file_name+".csv",keep_columns=["dataset1_id", "dataset2_id","matching_topic"]);
    df_not_matching = df_ds[df_ds["matching_topic"] == 0 ].to_numpy()
    df_matching = df_ds[df_ds["matching_topic"] == 1 ].to_numpy()

    np.random.shuffle(df_not_matching)
    np.random.shuffle(df_matching)

    #split for test and training
    pos_train = df_matching[0:int(len(df_matching)*0.8)]
    pos_test = df_matching[len(pos_train):]
    neg_train = df_not_matching[0:int(len(df_not_matching)*0.8)]
    neg_test = df_not_matching[len(neg_train):]
    train = concat_shuffle(pos_train,neg_train)
    test = concat_shuffle(pos_test,neg_test)
    
    #sampling neg/pos ratio + data augmentation
    if neg_sample > 0:
        train = sample_negative(train,neg_sample)
    test = sample_negative(test,1)
    train = data_augmentation(train)
    
    path = file_name+"/random/"+str(neg_sample)
    write_files(path,train,test)
    print("Train/Test split done")
    return path

strategy = ["isolation","random"]
def split_ds(dataset_name,s,neg_sampling,cv=False):
    if s == strategy[0]:
        if cv:
            return split_cv_isolation(dataset_name,neg_sampling)
        else:
            return split_isolation(dataset_name,neg_sampling)
    if s == strategy[1]:
        if cv:
            return split_cv_random(dataset_name,neg_sampling)
        else:
            return split_random(dataset_name,neg_sampling)