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

def get_splits(data,splits=14):
    kf = KFold(n_splits=splits,shuffle=True)
    neg_samples = []
    i = []
    #create one split and return - for now, just 1 fold for experiments
    for train_index, test_index in kf.split(data):
        _train = data[train_index]
        _test = data[test_index]
        break
    return _train,_test

def concat_shuffle(a1,a2):
    out = np.concatenate((a1,a2)) 
    np.random.shuffle(out)
    return out

#control there are not overlapping pairs
def overlapping_pairs(data):
    control = []
    for r in data:
        comb1 = str("{}_{}".format(r[0],r[1]))
        comb2 = str("{}_{}".format(r[1],r[0]))
        if comb1 in control or comb2 in control:
            return True
        else:
            control.append(comb1)
            control.append(comb2)
    return False
    
def split_ds2(file_name):
    df_ds = read_dataset("./datasets/"+file_name,keep_columns=["dataset1_id", "dataset2_id","matching_topic"]);
    df_not_matching = df_ds[df_ds["matching_topic"] == 0 ].to_numpy()
    df_matching = df_ds[df_ds["matching_topic"] == 1 ].to_numpy()

    np.random.shuffle(df_not_matching)
    np.random.shuffle(df_matching)
    
    df_not_matching = df_not_matching[0:len(df_matching)*2]
    
    pos_samples_concat = np.concatenate((df_matching[:,0],df_matching[:,1]))
    pos_samples_indices = np.unique(pos_samples_concat)
    pos_samples_indices_len = len(pos_samples_indices)
    
    train_pos_samples_indices_len = int(pos_samples_indices_len * 0.8)
    test_pos_samples_indices_len = pos_samples_indices_len - train_pos_samples_indices_len
    
    train_pos_indices = pos_samples_indices[0:train_pos_samples_indices_len-1]
    test_pos_indices = pos_samples_indices[train_pos_samples_indices_len: -1]
    
    
    train_pos = np.array([x for x in df_matching if x[0] in train_pos_indices and x[1] in train_pos_indices])
    train_neg = np.array([x for x in df_not_matching if x[0] in train_pos_indices and x[1] in train_pos_indices])
    
    test_pos = np.array([x for x in df_matching if x[0] in test_pos_indices and x[1] in test_pos_indices])
    test_neg = np.array([x for x in df_not_matching if x[0] in test_pos_indices and x[1] in test_pos_indices])
    
    train = np.concatenate((train_pos,train_neg))
    test = np.concatenate((test_pos,test_neg))
    
    dif_pos_neg = int (len(train_neg) / len(train_pos))
#     print("Ration neg / pos in train: " + str(dif_pos_neg))
    for i in range(dif_pos_neg):
        np.random.shuffle(train_pos)
        train = np.concatenate((train_pos,train))
        
    train_pos = np.array([x for x in train if x[2] == 1])
    np.random.shuffle(train)
    
    #write files
    df_train = pd.DataFrame(data=train,columns=["dataset1_id","dataset2_id","matching_topic"]) 
    df_test = pd.DataFrame(data=test,columns=["dataset1_id","dataset2_id","matching_topic"]) 
    df_train.to_csv("./datasets/"+file_name+"_train2.csv",index=False)
    df_test.to_csv("./datasets/"+file_name+"_test2.csv",index=False)
    
    print("Train/Test split done")

    
def split_topic(file_name):
    df_ds = read_dataset("./datasets/"+file_name,keep_columns=["dataset1_id", "dataset2_id","matching_topic","topic"]).to_numpy()
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
    
    #load pairs having the previous separated node-id in test, the rest in train  
    test = []
    train = []
    for pair in df_ds:
        if pair[0] in topic_pos_test or pair[1] in topic_pos_test:
            test.append(pair)
        else:
            train.append(pair)
    
    test = np.array(test)
    train = np.array(train)
    
    ###data augmentation    
    train_pos = np.array([x for x in train if x[2] == 1])
    train_neg = np.array([x for x in train if x[2] == 0])
    
    #Just 2x of negatives / positives. Comment to use all negatives
    np.random.shuffle(train_neg)
    train_neg = train_neg[0:len(train_pos)*2]
    train = np.concatenate((train_pos,train_neg))
    dif_pos_neg = int (len(train_neg) / len(train_pos))
    for i in range(dif_pos_neg-1):
        np.random.shuffle(train_pos)
        train = np.concatenate((train_pos,train))
    
    np.random.shuffle(train)
    np.random.shuffle(test)
    
    #write files
    df_train = pd.DataFrame(data=train[:,:3],columns=["dataset1_id","dataset2_id","matching_topic"]) 
    df_test = pd.DataFrame(data=test[:,:3],columns=["dataset1_id","dataset2_id","matching_topic"]) 
    df_train.to_csv("./datasets/"+file_name+"_train2x.csv",index=False)
    df_test.to_csv("./datasets/"+file_name+"_test2x.csv",index=False)
    
    print("Train/Test split done")
    
    
#     #print total ids number with possitive pairs 
#     ids_pos = np.unique(np.concatenate((df_matching[:,0],df_matching[:,1])))
#     print("\n")
#     print(ids_pos)
#     print("Number of ids in DATASET possitive pairs "+str(len(ids_pos)))
    
#     train_pos = np.array([x for x in train if x[2] == 1])
#     #print total ids number with possitive train pairs
#     ids_pos = np.unique(np.concatenate((train_pos[:,0],train_pos[:,1])))
#     print("\n")
#     print(ids_pos)
#     print("Number of ids in TRAIN possitive pairs "+str(len(ids_pos)))
    
#     test_pos = np.array([x for x in test if x[2] == 1])
#     #print total ids number with possitive test pairs
#     ids_pos = np.unique(np.concatenate((test_pos[:,0],test_pos[:,1])))
#     print("\n")
#     print(ids_pos)
#     print("Number of ids in TEST possitive pairs "+str(len(ids_pos)))
    
#     print("Train size: " + str(len(train)))
#     print("Train_positive size: " + str(len(train_pos)))
#     print("Test size: " + str(len(test)))
#     print("Test_positive size: " + str(len(test_pos)))
    
    
    
def split_ds(file_name):
    df_ds = read_dataset("./datasets/"+file_name,keep_columns=["dataset1_id", "dataset2_id","matching_topic"]);
    df_not_matching = df_ds[df_ds["matching_topic"] == 0 ].to_numpy()
    df_matching = df_ds[df_ds["matching_topic"] == 1 ].to_numpy()

    np.random.shuffle(df_not_matching)
    np.random.shuffle(df_matching)
    
    #Sample 2x negative per positive pair
    neg_sample = df_not_matching[0:len(df_matching)*2]

    #1 split for test, 5 for training
    neg_train,neg_test = get_splits(neg_sample,6)
    pos_train,pos_test = get_splits(df_matching,6)

    #concat negative and possitive pairs splits
    train = concat_shuffle(pos_train,neg_train)
    test = concat_shuffle(pos_test,neg_test)

    train_matching = np.array([x for x in train if x[2]==1])
    train_not_matching = np.array([x for x in train if x[2]==0])
    test_matching = np.array([x for x in test if x[2]==1])
    test_not_matching = np.array([x for x in test if x[2]==0])

    #data augmentation for positive data
    train_matching.T[[0, 1]] = train_matching.T[[1, 0]]
    train=np.concatenate((train,train_matching))
    np.random.shuffle(train)
    train_matching = np.array([x for x in train if x[2]==1])
    train_not_matching = np.array([x for x in train if x[2]==0])

    #write files
    df_train = pd.DataFrame(data=train,columns=["dataset1_id","dataset2_id","matching_topic"]) 
    df_test = pd.DataFrame(data=test,columns=["dataset1_id","dataset2_id","matching_topic"]) 
    df_train.to_csv("./datasets/"+file_name+"_train.csv",index=False)
    df_test.to_csv("./datasets/"+file_name+"_test.csv",index=False)
    
    print("Train/Test split done")


