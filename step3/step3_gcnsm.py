#!/usr/bin/env python
# coding: utf-8

# # Get dataset with ~80% train, ~20% test

# In[ ]:


import numpy as np
import pandas as pd
from step3 import step3_train_test_split as ds_split

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
        
    if ns == None or not int(ns) or ns < 0: 
        parameter_error("neg_sample",neg_sample)
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
    if we == None or not str(we) or we not in ["BERT","FASTTEXT","FASTTEXT2"]:
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


# # Read graph of metafeatures

# In[ ]:


import networkx as nx 
map_ds = None
map_reverse_ds_order = None
def load_graph():
    global map_ds
    global map_reverse_ds_order
    
    if word_embedding_encoding == "FASTTEXT":
        g_x = nx.read_gpickle("./word_embeddings/encoded_fasttext.gpickle")
    if word_embedding_encoding == "FASTTEXT2":
        g_x = nx.read_gpickle("./word_embeddings/encoded_fasttext_v2.gpickle")    
    if word_embedding_encoding == "BERT":
        g_x = nx.read_gpickle("./word_embeddings/encoded_bert.gpickle")

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

    datasets = [x for (x,y) in g_x.nodes(data=True) if y['tipo']==0]
    ds_order = [y['ds_order'] for x,y in g_x.nodes(data=True) if y['tipo']==0]
    map_ds = dict(zip(datasets,ds_order))
    map_reverse_ds_order = dict(zip(ds_order,datasets))
    map_ds['DS_1']

    for mask in train_mask:
        mask[0] = map_ds["DS_"+str(mask[0])]
        mask[1] = map_ds["DS_"+str(mask[1])]
        if mask[2] == 0:
            mask[2] = -1
    for mask in test_mask:
        mask[0] = map_ds["DS_"+str(mask[0])]
        mask[1] = map_ds["DS_"+str(mask[1])]
    
    return g_x


# ### Export graph to deep graph library

# In[ ]:


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


# # Training

# ### Evaluation methods

# In[ ]:


# Accuracy based on thresholds of distance (e.g. cosine > 0.8 should be a positive pair)
def threshold_acc(model, g, features, mask,loss,print_details=False,threshold_dist=0.2,threshold_cos=0.8):
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
                indices.append(0.0)          
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
                indices.append(0.0)
        indices_tensor = th.tensor(indices)
        labels_tensor = th.tensor(mask[:,2])
    
    if print_details:
        positives = 0.0
        negatives = 0.0
        true_positives = 0.0
        true_negatives = 0.0
        false_positives = 0.0
        false_negatives = 0.0
        
        for i in range(len(labels_tensor)):
            prediction = indices_tensor[i].item()
            label = labels_tensor[i].item()
            if label == 0.0:
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
        
        #print confusion matrix            
        print("\t \t \t \t ##########Labels##########")
        print("\t \t \t \t Similar \t Not Similar")
        print("Prediction Similar: \t \t {} \t \t {}".format(true_positives,false_positives))
        print("Prediction Not Similar:  \t {} \t \t {}".format(false_negatives,true_negatives))
        print("\t \t \t \t----------------------")
        print("\t \t \t \t{} \t \t {}".format(positives,negatives))
        print("\nRecall/Sensitivity: "+str(true_positives/positives))
        print("Specificity/Selectivity: "+str(true_negatives/negatives))
        print("Accuracy: "+str((true_positives + true_negatives) / len(labels_tensor)))
        return (true_positives + true_negatives) / len(labels_tensor)
    else:
        correct = th.sum(indices_tensor == labels_tensor)
        return correct.item() * 1.0 / len(labels_tensor)

# Accuracy based on nearest neighboor (e.g. the nearest node should be a positive pair)
def ne_ne_acc_isolation(model, g, features, mask,loss,print_details=False):

    train_indices = np.unique(np.concatenate((train_mask[:,0],train_mask[:,1])))
    train_pos_samples = np.array([x for x in train_mask if x[2]==1])    
    train_pos_samples_indices = np.unique(np.concatenate((train_pos_samples[:,0],train_pos_samples[:,1])))
    
    mask_indices = np.unique(np.concatenate((mask[:,0],mask[:,1])))
    mask_pos_samples = np.array([x for x in mask if x[2]==1])    
    mask_pos_samples_indices = np.unique(np.concatenate((mask_pos_samples[:,0],mask_pos_samples[:,1])))
    mask_pos_samples_indices = np.array([x for x in mask_pos_samples_indices if x not in train_pos_samples_indices ])
    
    pos_samples = np.concatenate((train_pos_samples,mask_pos_samples))
    pos_samples_indices = np.unique(np.concatenate((pos_samples[:,0],pos_samples[:,1])))
    train_embeddings,mask_pos_samples_embeddings = model(g, features,train_pos_samples_indices,mask_pos_samples_indices)
    
    sum_accuracy = 0
    for i in range(len(mask_pos_samples_indices)):
        candidate = mask_pos_samples_embeddings[i]
        #dist() | m - dist()
        if loss == "ContrastiveLoss":
            pdist = th.nn.PairwiseDistance(p=2)        
            result = pdist(candidate,train_embeddings)
            largest = False
        #1 - cos() | max(0,cos() - m)
        if loss == "CosineEmbeddingLoss":
            thecos = th.nn.CosineSimilarity(dim=1, eps=1e-6)
            result = thecos(candidate.reshape(1,len(candidate)),train_embeddings)
            largest = True
        
        #we ignore the result of the vector with itself
#         print("Candidate id: " + str(mask_pos_samples_indices[i]))        
        result_indices = th.topk(result, 2, largest=largest).indices
        closest_node_index = th.tensor(train_pos_samples_indices)[result_indices]
#         print(closest_node_index)
#         print("all in mask")
#         print(mask_pos_samples)
        
#         check_relation_nodes = np.array([x for x in pos_samples 
        check_relation_nodes = np.array([x for x in mask_pos_samples 
                                         if (x[0]==mask_pos_samples_indices[i] and x[1] in closest_node_index) or 
                                         (x[1]==mask_pos_samples_indices[i] and x[0] in closest_node_index)])
#         print("relations found: ")
#         print(check_relation_nodes)
        if len(check_relation_nodes) > 0:
            sum_accuracy += 1
    
    return sum_accuracy / len(mask_pos_samples_indices)    

# Accuracy based on nearest neighboor (e.g. the nearest node should be a positive pair)
def ne_ne_acc_random(model, g, features, mask,loss,print_details=False):

    train_indices = np.unique(np.concatenate((train_mask[:,0],train_mask[:,1])))
    train_pos_samples = np.array([x for x in train_mask if x[2]==1])    
    train_pos_samples_indices = np.unique(np.concatenate((train_pos_samples[:,0],train_pos_samples[:,1])))
    
    mask_indices = np.unique(np.concatenate((mask[:,0],mask[:,1])))
    mask_pos_samples = np.array([x for x in mask if x[2]==1])    
    mask_pos_samples_indices = np.unique(np.concatenate((mask_pos_samples[:,0],mask_pos_samples[:,1])))
#     mask_pos_samples_indices = np.array([x for x in mask_pos_samples_indices if x not in train_pos_samples_indices ])
    
    pos_samples = np.concatenate((train_pos_samples,mask_pos_samples))
    pos_samples_indices = np.unique(np.concatenate((pos_samples[:,0],pos_samples[:,1])))
    pos_embeddings,mask_pos_samples_embeddings = model(g, features,pos_samples_indices,mask_pos_samples_indices)
    
    sum_accuracy = 0
    for i in range(len(mask_pos_samples_indices)):
        candidate = mask_pos_samples_embeddings[i]
        #dist() | m - dist()
        if loss == "ContrastiveLoss":
            pdist = th.nn.PairwiseDistance(p=2)        
            result = pdist(candidate,pos_embeddings)
            largest = False
        #1 - cos() | max(0,cos() - m)
        if loss == "CosineEmbeddingLoss":
            thecos = th.nn.CosineSimilarity(dim=1, eps=1e-6)
            result = thecos(candidate.reshape(1,len(candidate)),pos_embeddings)
            largest = True
        
        #we ignore the result of the vector with itself
#         print("Candidate id: " + str(mask_pos_samples_indices[i]))        
        result_indices = th.topk(result, 2, largest=largest).indices
        closest_node_index = th.tensor(pos_samples_indices)[result_indices]
#         print(closest_node_index)
#         print("all in mask")
#         print(mask_pos_samples)
        
#         check_relation_nodes = np.array([x for x in pos_samples 
        check_relation_nodes = np.array([x for x in pos_samples 
                                         if (x[0]==mask_pos_samples_indices[i] and x[1] in closest_node_index) or 
                                         (x[1]==mask_pos_samples_indices[i] and x[0] in closest_node_index)])
#         print("relations found: ")
#         print(check_relation_nodes)
        if len(check_relation_nodes) > 0:
            sum_accuracy += 1
    
    return sum_accuracy / len(mask_pos_samples_indices)    

def confusion_matrix(model, g, features, mask,loss,threshold):
    model.eval()
    with th.no_grad():
        acc = threshold_acc(model, g, features, mask,loss,print_details=True,threshold_dist=threshold,threshold_cos=threshold)
        return acc
        
def evaluate(model, g, features, mask,loss):
    model.eval()
    with th.no_grad():
        #naive way of testing accuracy 
        acc = threshold_acc(model, g, features, mask,loss)
        #accuracy based on 1-NN 
        if strategy == "isolation":
            acc2 = ne_ne_acc_isolation(model, g, features, mask,loss)
        if strategy == "random":
            acc2 = ne_ne_acc_random(model, g, features, mask,loss)
        return acc,acc2


# ### Train loop

# In[ ]:


import time
import numpy as np
def train(training,iterations):
    dur = []
    
    #set max accuracy found if model already has state
    max_acc = 0.0
    max_acc2 = 0.0
    if len(training.log) > 0:
        for l in training.log:
            if l["acc"] > max_acc:
                max_acc = l["acc"]
                max_acc2 = l["acc2"]
            else:        
                if l["acc"] == max_acc and l["acc2"] > max_acc2:
                    max_acc2 = l["acc2"]
            
    not_improving = 0
    ## create batchs for training
    numb_splits = int(len(train_mask) / training.batch_splits) + 1
    train_batch = np.array_split(train_mask,numb_splits)
    
    #specify number of threads for the training
    #th.set_num_threads(2)
    
    for epoch in range(iterations):
        #model train mode
        training.net.train()
        t0 = time.time()
        epoch_loss = 0
        
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
        acc,acc2 = evaluate(training.net, g, g.ndata['vector'], test_mask,training.loss_name)
        
        #create log
        output = {}
        output['epoch'] = training.epochs_run
        output['loss'] = float('%.5f'% (epoch_loss))
        output['acc'] = float('%.5f'% (acc))
        output['acc2'] = float('%.5f'% (acc2))
        output['time_epoch'] = float('%.5f'% (np.mean(dur)))
        output['time_total'] = float('%.5f'% (training.runtime_seconds))
        training.log.append(output)
        training.epochs_run+=1
        print(str(output))
        
        ##save best model and results found so far
        if acc > max_acc:
            print("Best model found so far...")
            training.set_best(training)
            max_acc = acc
            max_acc2 = acc2
            not_improving = 0
        else:        
            if acc == max_acc and acc2 > max_acc2:
                print("Best model found so far...")
                training.set_best(training)
                max_acc = acc
                max_acc2 = acc2
                not_improving = 0
            #if not improvments for 30 epochs in a row, then stop    
            else:
                if not_improving < 100:
                    not_improving +=1
                else:
                    print("Not improving anymore...finishing training.")
                    pad = iterations - epoch -1
                    training.epochs_run+=pad
                    break
                    
                 
    #save final model state and final results
    training.save_state(path_setup)



import copy
import os
from pathlib import Path


# cv_logs = []
# def cv_training(training,it):
#     train(training,iterations=it)
#     return training.log

def cross_validation(training,iterations=1,ran="1-10",nsample=None,create=None):
    global cv_logs
    
    if nsample == None:
        nsample = neg_sample
    if create == None:
        create = create_new_split
    
    cv_ran = ran.split("-")
    init = int(cv_ran[0]) -1
    ending = int(cv_ran[1])
    if init < 0 or ending >10:
        print("Out of range: CV is from 1 to 10")
    
    
    training_copy = None
    for i in range(init,ending):
        load_env(ds_name=dataset_name,ns=nsample,st=strategy,sp=create,we=word_embedding_encoding,cv=i)
        training_copy = copy.deepcopy(training)
        train(training_copy,iterations)
        path_setup = dataset_name+"/"+strategy+"/"+str(neg_sample)+"/cv"
        training_copy.save_state(path_setup,"/tmp_cv_result_"+str(i))