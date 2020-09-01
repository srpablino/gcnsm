import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import os    
from pathlib import Path

def concat_shuffle(a1,a2):
    out = np.concatenate((a1,a2)) 
    np.random.shuffle(out)
    return out

def sample_negative(triple_list,n=1):
    np.random.shuffle(triple_list)
    t_pos = np.array([x for x in triple_list if x[2]==1])
    t_neg = np.array([x for x in triple_list if x[2]==0])
    n = min(n,int(len(t_neg)/len(t_pos)))
    np.random.shuffle(t_neg)
    t_neg = t_neg[0:len(t_pos)*n]
    result = np.concatenate((t_pos,t_neg))
    return result
 
def write_files(outdir,train,test):
    #write files
    df_train = pd.DataFrame(data=train,columns=["id_1","id_2","match"]) 
    df_test = pd.DataFrame(data=test,columns=["id_1","id_2","match"]) 
    if not os.path.exists(outdir):
        Path(outdir).mkdir(parents=True, exist_ok=True)
    df_train.to_csv(outdir+"/train.csv",index=False)
    df_test.to_csv(outdir+"/test.csv",index=False)
    
def random_subsam(dataset_name):
    outputdir = dataset_name+"/random_subsam"
    for i in range(20):
        print("ITERATION: "+str(i))
        df_ds = pd.read_csv("./ground_truth/"+dataset_name+"/"+dataset_name+".csv").to_numpy();
        df_matching = np.array([x for x in df_ds if x[2] == 1])
        np.random.shuffle(df_matching)

        #For each topic with possitive pairs, separate the 20% of node_ids in test
        topics = np.unique(df_ds[:,3])
        topic_pos_test = []
        topic_pos_train = []
        for t in topics:
            topic_pos_pairs = np.array([x for x in df_matching if x[3] == t])
            if len(topic_pos_pairs) == 0:
                continue
            topic_pos_ds = np.unique(np.concatenate((topic_pos_pairs[:,0],topic_pos_pairs[:,1])))
            if len(topic_pos_ds) > 5:
                np.random.shuffle(topic_pos_ds)
                topic_pos_test.append(topic_pos_ds[0])
                for n in range(1,len(topic_pos_ds)):
                    topic_pos_train.append(topic_pos_ds[n])
            else:
                for n in range(len(topic_pos_ds)):
                    topic_pos_train.append(topic_pos_ds[n])
                

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

        test = sample_negative(test)

        write_files("./ground_truth/"+outputdir+"/"+str(i),train[:,:3],test[:,:3])
    print("Train/Test split done")
    return outputdir

def cv_10(dataset):
    df_ds = pd.read_csv("./ground_truth/"+dataset+"/"+dataset+".csv").to_numpy();
    
    df_not_matching = np.array([x for x in df_ds if x[2]==0])
    df_matching = np.array([x for x in df_ds if x[2]==1])
    np.random.shuffle(df_matching)
    np.random.shuffle(df_not_matching)

    cv_pos = np.array(np.array_split(df_matching,10))
    cv_neg = np.array(np.array_split(df_not_matching,10))
    output_path = dataset+"/10_cv/"
    for i in range(10):
        test = np.concatenate((cv_pos[i],cv_neg[i]))
        
        train_pos = np.concatenate((cv_pos[0:i],cv_pos[i+1:]))
        train_neg = np.concatenate((cv_neg[0:i],cv_neg[i+1:]))
        
        train_pos = np.concatenate((train_pos))
        train_neg = np.concatenate((train_neg))
        
        train = np.concatenate((train_pos,train_neg)).squeeze()
        
        test = sample_negative(test,1)
        
        write_files("./ground_truth/"+output_path+"/"+str(i),train[:,:3],test[:,:3])
            
    
    print("CV Train/Test split done")
    return output_path

experiments = ["random_subsam","10_cv"]
def split_ds(dataset_name,experiment):
    if experiment == experiments[0]:
        return random_subsam(dataset_name)
    if experiment == experiments[1]:
        return cv_10(dataset_name)