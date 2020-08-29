## Get dataset with ~80% train, ~20% test
import numpy as np
import pandas as pd
from step3 import step3_train_test_split as ds_split
import copy
import os
from pathlib import Path
import time

# default values
train_mask = None
test_mask = None
neg_sample = 2
strategy = "random"
create_new_split = False
word_embedding_encoding = "FASTTEXT"
path_setup = None
dataset_name = "openml_203ds_datasets_matching"
cross_v=-1

def parameter_error(param_error,value):
    print("Encounter error in parameter {}, default value: {} will be used ".format(param_error,value))
    
def load_env(ds_name=None,ns=None,st=None,sp=None,we=None,cv=-1): 
    global dataset_name
    global neg_sample
    global strategy
    global create_new_split
    global word_embedding_encoding
    global train_mask
    global test_mask
    global path_setup
    global cross_v
    
    cross_v = cv

    #see dataset_setup to config parametrs for split data
    if ds_name == None or not str(ds_name): 
        parameter_error("dataset_name",dataset_name)
    else:
        dataset_name = ds_name
    
    if ns == None or ns < 0: 
        parameter_error("neg_sample: "+str(ns),neg_sample)
    else:
        neg_sample = ns
        
    if st == None or not str(st) or st not in ["isolation","random"]:
        parameter_error("strategy",strategy)
    else:
        strategy = st
    if sp == None:
        parameter_error("create_new_split",create_new_split)
    else:
        create_new_split = sp
    if we == None or not str(we) or we not in ["BERT","BERT2","FASTTEXT","FASTTEXT2","FASTTEXT2_2","FASTTEXT2_NAMES","FASTTEXT2_SHORT","FASTTEXT_SIMPLE_CLEAN","FASTTEXT2_CLEAN","FASTTEXT2_CLEAN5","FASTTEXT2_CLEAN4","FASTTEXT2_CLEAN3","FASTTEXT2_CLEAN2","FASTTEXT2_NEW","FASTTEXT_SIMPLE","FASTTEXT_SIMPLE_NAMES","FASTTEXT_SIMPLE_SHORT","MONITOR_SIMPLE","MONITOR_SIMPLE_SHORT","MONITOR_CLEAN","MONITOR_CLEAN2"]:
        parameter_error("word_embedding_encoding",word_embedding_encoding)
    else:
        word_embedding_encoding = we

    print("Values to load")
    print("dataset_name="+dataset_name)
    print("neg_sample= "+str(neg_sample))
    print("strategy= "+strategy)
    print("create_new_split= "+str(create_new_split))
    print("word_embedding_encoding= "+word_embedding_encoding)    
    print("cross_v= "+str(cross_v))    

    if cross_v < 0:
        if create_new_split:
            print("Creating simple train/test splits...")
            path_setup = ds_split.split_ds(dataset_name,strategy,neg_sample)
        else:
            path_setup = dataset_name+"/"+strategy+"/"+str(neg_sample)
        
        train_mask = pd.read_csv("./datasets/"+path_setup+"/train.csv").to_numpy()
        test_mask = pd.read_csv("./datasets/"+path_setup+"/test.csv").to_numpy()
    
    else:
        if cross_v == 0 and create_new_split:
            print("Creating cross validation splits...")
            path_setup = ds_split.split_ds(dataset_name,strategy,neg_sample,True)
        else:
            path_setup = dataset_name+"/"+strategy+"/"+str(neg_sample)+"/cv"
            
        
        train_mask = pd.read_csv("./datasets/"+path_setup+"/"+str(cross_v)+"/train.csv").to_numpy()
        test_mask = pd.read_csv("./datasets/"+path_setup+"/"+str(cross_v)+"/test.csv").to_numpy()
    
    

    #info about split
    train_positive = np.array([x for x in train_mask if x[2]==1])
    test_positive = np.array([x for x in test_mask if x[2]==1])
    print("Dataset splits loaded")
    print("Train samples: "+str(len(train_mask)) + " Test samples: "+str(len(test_mask)))
    print("Train positive samples: "+str(len(train_positive)) + " Test positive samples: "+str(len(test_positive)))
    load_dgl()


## Read graph of metafeatures
import networkx as nx 
map_ds = None
map_reverse_ds_order = None
def load_graph():
    global map_ds
    global map_reverse_ds_order
    global train_mask
    global test_mask
    
    if word_embedding_encoding == "FASTTEXT":
        g_x = nx.read_gpickle("./word_embeddings/encoded_fasttext.gpickle")
    if word_embedding_encoding == "FASTTEXT_SIMPLE":
        g_x = nx.read_gpickle("./word_embeddings/encoded_fasttext_simple.gpickle")    
    if word_embedding_encoding == "FASTTEXT_SIMPLE_NAMES":
        g_x = nx.read_gpickle("./word_embeddings/encoded_fasttext_simple_names.gpickle")    
    if word_embedding_encoding == "FASTTEXT_SIMPLE_SHORT":
        g_x = nx.read_gpickle("./word_embeddings/encoded_fasttext_simple_short.gpickle")  
    if word_embedding_encoding == "FASTTEXT_SIMPLE_CLEAN":
        g_x = nx.read_gpickle("./word_embeddings/clean_fasttext_simple_short.gpickle")        
        
    if word_embedding_encoding == "FASTTEXT2":
        g_x = nx.read_gpickle("./word_embeddings/encoded_fasttext_v2.gpickle")    
    if word_embedding_encoding == "FASTTEXT2_2":
        g_x = nx.read_gpickle("./word_embeddings/encoded_fasttext2.gpickle")        
    if word_embedding_encoding == "FASTTEXT2_NAMES":
        g_x = nx.read_gpickle("./word_embeddings/encoded_fasttext2_names.gpickle")    
    if word_embedding_encoding == "FASTTEXT2_SHORT":
        g_x = nx.read_gpickle("./word_embeddings/encoded_fasttext2_short.gpickle")    
    
    if word_embedding_encoding == "FASTTEXT2_NEW":
        g_x = nx.read_gpickle("./word_embeddings/new_fasttext_short.gpickle")    
    
    if word_embedding_encoding == "FASTTEXT2_CLEAN":
        g_x = nx.read_gpickle("./word_embeddings/clean_fasttext_short.gpickle")    
    if word_embedding_encoding == "FASTTEXT2_CLEAN2":
        g_x = nx.read_gpickle("./word_embeddings/clean2_fasttext_short.gpickle")    
    if word_embedding_encoding == "FASTTEXT2_CLEAN3":
        g_x = nx.read_gpickle("./word_embeddings/clean3_fasttext_short.gpickle")    
    if word_embedding_encoding == "FASTTEXT2_CLEAN4":
        g_x = nx.read_gpickle("./word_embeddings/clean4_fasttext_short.gpickle")    
    if word_embedding_encoding == "FASTTEXT2_CLEAN5":
        g_x = nx.read_gpickle("./word_embeddings/clean5_fasttext_short.gpickle")    
    
    if word_embedding_encoding == "MONITOR_SIMPLE":
        g_x = nx.read_gpickle("./word_embeddings/monitor_fasttext_simple.gpickle")   
    if word_embedding_encoding == "MONITOR_SIMPLE_SHORT":
        g_x = nx.read_gpickle("./word_embeddings/monitor_fasttext_simple_short.gpickle")   
        
    if word_embedding_encoding == "MONITOR_CLEAN":
        g_x = nx.read_gpickle("./word_embeddings/clean_monitor_fasttext_short.gpickle")   
    if word_embedding_encoding == "MONITOR_CLEAN2":
        g_x = nx.read_gpickle("./word_embeddings/clean_monitor_fasttext_short2.gpickle")   
        
    if word_embedding_encoding == "BERT":
        g_x = nx.read_gpickle("./word_embeddings/encoded_bert.gpickle")
    if word_embedding_encoding == "BERT_SIMPLE":
        g_x = nx.read_gpickle("./word_embeddings/encoded_bert_simple.gpickle")    
        
    if word_embedding_encoding == "BERT2":
        g_x = nx.read_gpickle("./word_embeddings/new_bert_short.gpickle")    
    

    ds_order = 0
    for x,n in sorted(g_x.nodes(data=True)):
        t = n['tipo']
        if t == "dataset":
            n['tipo'] = 0
        if t == "feature dataset":
            n['tipo'] = 1
        if t == "literal dataset":
            n['tipo'] = 2
        if t == "attribute":
            n['tipo'] = 3
        if t == "feature attribute":
            n['tipo'] = 4
        if t == "literal attribute":
            n['tipo'] = 5  
        n['ds_order']=ds_order
        ds_order+=1

    datasets = [x.strip() for (x,y) in g_x.nodes(data=True) if y['tipo']==0]
    ds_order = [y['ds_order'] for x,y in g_x.nodes(data=True) if y['tipo']==0]
    map_ds = dict(zip(datasets,ds_order))
    map_reverse_ds_order = dict(zip(ds_order,datasets))

    for mask in train_mask:
        mask[0] = map_ds["DS_"+str(mask[0]).strip().replace('\xa0','')]
        mask[1] = map_ds["DS_"+str(mask[1]).strip().replace('\xa0','')]            
        if mask[2] == 0:
            mask[2] = -1
            
    for mask in test_mask:
        mask[0] = map_ds["DS_"+str(mask[0]).strip().replace('\xa0','')]
        mask[1] = map_ds["DS_"+str(mask[1]).strip().replace('\xa0','')]
        if mask[2] == 0:
            mask[2] = -1
        
    
    train_mask = train_mask.astype(np.float) 
    test_mask = test_mask.astype(np.float) 
    
    return g_x


## Export graph to deep graph library
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
#convert from networkx to graph deep library format
g = None
def load_dgl():
    global g
    g_x = load_graph()
    g = dgl.DGLGraph()
    g.from_networkx(g_x,node_attrs=['tipo','vector','ds_order'], edge_attrs=None)
    print("Meta-feature graph from datasets loaded")


# Accuracy based on thresholds of distance (e.g. cosine > 0.8 should be a positive pair)
def threshold_acc(model, g, features, mask,loss,print_details=False,threshold_dist=0.2,threshold_cos=0.5,path=None):
    indices = []
    
    #mask = np.array([x for x in mask if x[2]==1])
    
    z1, z2 = model(g,features,mask[:,0],mask[:,1])
    
    #dist() | max(0, m - dist())
    if loss == "ContrastiveLoss" or loss == "Euclidean":
        pdist = th.nn.PairwiseDistance(p=2)        
        result = pdist(z1,z2)
        for i in range(len(result)):
            r = result[i]
            if r.item() <= threshold_dist:
                indices.append(1.0)
            else:
                indices.append(-1.0)          
        indices_tensor = th.tensor(indices)
        labels_tensor = th.tensor(mask[:,2])
        
    #1 - cos() | max(0,cos() - m)
    if loss == "CosineEmbeddingLoss":
        cos = th.nn.CosineSimilarity(dim=1, eps=1e-6)
        result = cos(z1,z2)
        for i in range(len(result)):
            r = result[i]
            if r.item() >= threshold_cos:
                indices.append(1.0)
            else:
                indices.append(-1.0)
        indices_tensor = th.tensor(indices)
        labels_tensor = th.tensor(mask[:,2])
    
    positives = 0.0
    negatives = 0.0
    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    for i in range(len(labels_tensor)):
        prediction = indices_tensor[i].item()
        label = labels_tensor[i].item()
        if label == -1.0:
            negatives+=1
            if prediction == label:
                true_negatives+=1
            else:
                false_positives+=1
        else:
            positives+=1
            if prediction == label:
                true_positives+=1
            else:
                false_negatives+=1
        
    output = {}
    output['true_positives'] = true_positives
    output['false_positives'] = false_positives
    output['true_negatives'] = true_negatives
    output['false_negatives'] = false_negatives
    output['recall'] = true_positives/positives
    output['specificity'] = true_negatives/negatives
    output['precision'] = 0
    output['fscore'] = 0
    try:
        output['precision'] = true_positives / (true_positives + false_positives)
        output['fscore'] = 2 * (output['precision'] * output['recall']) / ((output['precision'] + output['recall']))
    except:
        print("precision and fscore not calculated")
    output['acc'] = (true_positives + true_negatives) / len(labels_tensor)
    
    if print_details:
        #print confusion matrix            
        print("\t \t \t \t ##########Labels##########")
        print("\t \t \t \t Similar \t Not Similar")
        print("Prediction Similar: \t \t {} \t \t {}".format(true_positives,false_positives))
        print("Prediction Not Similar:  \t {} \t \t {}".format(false_negatives,true_negatives))
        print("\t \t \t \t----------------------")
        print("\t \t \t \t{} \t \t {}".format(positives,negatives))
        print("\nRecall/Sensitivity: "+str(output['recall']))
        print("Precision: "+str(output['precision']))
        print("Fscore: "+str(output['fscore']))
    
    return output

def confusion_matrix(model, g, features, mask,loss,threshold):
    model.eval()
    for m in mask:
        if m[2]==0:
            m[2]=-1
    with th.no_grad():
        acc = threshold_acc(model, g, features, mask,loss,print_details=True,threshold_dist=threshold,threshold_cos=threshold)
        return acc
    
def confusion_matrix_cv(model, g, features, path,loss,threshold):
    model.eval()
    cv_number = path.split("tmp_cv_result_")[1].split(".")[0]
    path=path.split("/net_name")[0].replace("models/","datasets/")+"/"+cv_number
    tmp_test = pd.read_csv(path+"/test.csv").to_numpy()
    for mask in tmp_test:
        mask[0] = map_ds["DS_"+str(mask[0])]
        mask[1] = map_ds["DS_"+str(mask[1])]
        if mask[2]==0:
            mask[2]=-1
        
    with th.no_grad():
        acc = threshold_acc(model, g, features, tmp_test,loss,print_details=True,threshold_dist=threshold,threshold_cos=threshold)
        return acc    
    
def evaluate(training, g, features, mask,loss):
    training.net.eval()
    with th.no_grad():
        #test accuracy with threshold
        th_output = threshold_acc(training.net, g, features, mask,loss)
        
        #calculate test_loss        
        z1,z2 = training.net(g, g.ndata['vector'],mask[:,0],mask[:,1])
        loss_test = training.loss(z1,z2, th.tensor(mask[:,2]))

        return th_output,0,loss_test.item()

def shuffle_splits_ns(train_mask, n,ns=4):
    train_pos = np.array([x for x in train_mask if x[2]==1])
    train_neg = np.array([x for x in train_mask if x[2]==-1])
    nodes_pos = np.unique(np.concatenate((train_pos[:,0],train_pos[:,1])))
    result = []
    partial_results_neg = np.array([]).reshape(0,3)
    partial_results_pos = np.array([]).reshape(0,3)
    for node in nodes_pos:
        filter_node_pos = np.array([x for x in train_pos if x[0]==node or x[1]==node])
        ns_len =int((ns/2) * len(filter_node_pos))
        filter_node_neg = np.array([x for x in train_neg if x[0]==node or x[1]==node])
        np.random.shuffle(filter_node_neg)
        partial_results_neg = np.concatenate((partial_results_neg,filter_node_neg[0:ns_len]))
    
    #data aug
#     for i in range(ns):
#         partial_results_pos = np.concatenate((partial_results_pos,train_pos))
    
    #no data aug
    partial_results_pos = np.concatenate((partial_results_pos,train_pos))
    
    np.random.shuffle(partial_results_pos)
    np.random.shuffle(partial_results_neg)
    
    numb_splits = int(( len(partial_results_pos) + len(partial_results_neg) ) / n) + 1
    pos_batch = np.array_split(partial_results_pos,numb_splits)
    neg_batch = np.array_split(partial_results_neg,numb_splits)
    
    result = []
    for j in range(numb_splits):
        result.append(np.concatenate((pos_batch[j],neg_batch[j])))
        np.random.shuffle(result[j])
#     for r in result:
#         print(len(r))
    return np.array(result)


def shuffle_splits(train_mask, n):
    train_pos = [x for x in train_mask if x[2]==1]
    train_neg = [x for x in train_mask if x[2]==-1]
    np.random.shuffle(train_pos)
    np.random.shuffle(train_neg)
    pos_batch = np.array_split(train_pos,n)
    neg_batch = np.array_split(train_neg,n)
    result = []
    for i in range(n):
        result.append(np.concatenate((pos_batch[i],neg_batch[i])))
        np.random.shuffle(result[i])
    return np.array(result)
        
def train(training,iterations,nsample=4):
    dur = []
    print(str("Start of training...NN {} Loss {} Split {}: ").format(training.net_name,training.loss_name,training.batch_splits))
    #set max accuracy found if model already has state
    max_acc = 0.0
    max_acc2 = 0.0
    loss_min = 99999999.9
    if len(training.log) > 0:
        for l in training.log:
            if l["fscore"] > max_acc:
                max_acc = l["fscore"]
                max_acc2 = l["acc2"]
            else:        
                if l["fscore"] == max_acc and l["acc2"] > max_acc2:
                    max_acc2 = l["acc2"]
            
    need_update = 0
    ##original values (take into account for saving the model)
    o_lr = training.lr
    o_splits = training.batch_splits
    
    #specify number of threads for the training
    #th.set_num_threads(2)

    for epoch in range(iterations):
        #model train mode
        training.net.train()
        t0 = time.time()
        epoch_loss = 0
        
        ## create batchs and shuffle data for training
        train_batch = shuffle_splits_ns(train_mask,training.batch_splits,nsample)
        
        #forward_backward positive batch sample
        for split in train_batch:
            z1,z2 = training.net(g, g.ndata['vector'],split[:,0],split[:,1])
            loss = training.loss(z1,z2, th.tensor(split[:,2]))
            
            training.optimizer.zero_grad()
            #loss.backward(retain_graph=True)
            loss.backward()
            training.optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss = epoch_loss / training.batch_splits

        #runtime
        t = time.time() - t0
        dur.append(t)
        
        #total time accumulation for this model
        training.runtime_seconds+=t
        
        #accuracy
        th_output,acc2,loss_test = evaluate(training, g, g.ndata['vector'], test_mask,training.loss_name)
        
        #create log
        output = {}
        output['epoch'] = training.epochs_run
        output['loss'] = epoch_loss
        output['loss_test'] = loss_test
        output['acc'] = th_output['acc']
        output['acc2'] = acc2
        
        output['true_positives'] = th_output['true_positives']
        output['false_positives'] = th_output['false_positives']
        output['true_negatives'] = th_output['true_negatives']
        output['false_negatives'] = th_output['false_negatives']
        output['recall'] = th_output['recall']
        output['specificity'] = th_output['specificity']
        output['precision'] = th_output['precision']
        output['fscore'] = th_output['fscore']
        
        output['time_epoch'] = float('%.5f'% (np.mean(dur)))
        output['time_total'] = float('%.5f'% (training.runtime_seconds))
        
        #updated parameters
        output['lr'] = training.lr
        output['batch_splits'] = training.batch_splits
        
        training.log.append(output)
        training.epochs_run+=1
        print(str("Ep:{}, loss:{:.5f}, loss_test:{:.5f}, lr:{:.2e}, fs:{:.5f} (r={:.3f},p={:.3f}),  time:{:.3f}, tt:{:.3f}").format(output['epoch'],output['loss'],output['loss_test'],output['lr'],output['fscore'],output['recall'],output['precision'],output['time_epoch'],output['time_total']))
        
        ##save best model and results found so far
        if output['fscore'] > max_acc:
            print("##########Best model found so far##########")
            training.set_best(training)
            max_acc = output['fscore']
            max_acc2 = acc2
        training.set_lr(training.lr * .99)       
        
                                    
    #Recover initial setup to save file
    training.lr = o_lr
    training.batch_splits = o_splits
    if training.best != None:
        training.best.lr = o_lr 
        training.best.batch_splits = o_splits
    
    #save final model state and final results if experiment is not a CV
    if cross_v < 0:
        if training.best != None:
            training.best.epochs_run = training.epochs_run
        training.save_state(path_setup)                        
        
def cross_validation(training,iterations=1,ran="1-10",nsample=None,create=None):
    global cv_logs
    
    if nsample == None:
        nsample = neg_sample
    if create == None:
        create = create_new_split
    
    cv_ran = ran.split("-")
    init = int(cv_ran[0]) -1
    ending = int(cv_ran[1])
    if init < 0 or (ending >100 and strategy=="isolation") or (ending >10 and strategy=="random"):
        raise Exception("Values for CV out of range")
    
    
    training_copy = None
    for i in range(init,ending):
        load_env(ds_name=dataset_name,ns=0,st=strategy,sp=create,we=word_embedding_encoding,cv=i)
        training_copy = copy.deepcopy(training)
        train(training_copy,iterations,nsample)
        path_setup = dataset_name+"/"+strategy+"/"+str(neg_sample)+"/cv"
        if training_copy.best != None:
            training_copy.best.epochs_run = training_copy.epochs_run
        training_copy.save_state(path_setup,"/tmp_cv_result_"+str(i))
