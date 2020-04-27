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

#     print("Training set")
#     print(len(train))
#     print("Training pos")
#     print(len(train_matching))
#     print("Training neg")
#     print(len(train_not_matching))

#     print("Test set")
#     print(len(test))
#     print("Test pos")
#     print(len(test_matching))
#     print("Test neg")
#     print(len(test_not_matching))


#     #train pairs will have repeated possitive pairs
#     print(overlapping_pairs(train))
#     print(overlapping_pairs(test))

    #write files
    df_train = pd.DataFrame(data=train,columns=["dataset1_id","dataset2_id","matching_topic"]) 
    df_test = pd.DataFrame(data=test,columns=["dataset1_id","dataset2_id","matching_topic"]) 
    df_train.to_csv("./datasets/"+file_name+"_train.csv",index=False)
    df_test.to_csv("./datasets/"+file_name+"_test.csv",index=False)
    
    print("Train/Test split done")


