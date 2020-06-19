## Get dataset with ~80% train, ~20% test
import numpy as np
import pandas as pd
from step3 import step3_train_test_split as ds_split
import copy
import os
from pathlib import Path

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
    if we == None or not str(we) or we not in ["BERT","BERT2","FASTTEXT","FASTTEXT2","FASTTEXT_SIMPLE"]:
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
    
    if word_embedding_encoding == "FASTTEXT":
        g_x = nx.read_gpickle("./word_embeddings/encoded_fasttext.gpickle")
    if word_embedding_encoding == "FASTTEXT2":
        g_x = nx.read_gpickle("./word_embeddings/encoded_fasttext_v2.gpickle")    
    if word_embedding_encoding == "FASTTEXT_SIMPLE":
        g_x = nx.read_gpickle("./word_embeddings/encoded_fasttext_simple.gpickle")    
    if word_embedding_encoding == "BERT":
        g_x = nx.read_gpickle("./word_embeddings/encoded_bert.gpickle")
    if word_embedding_encoding == "BERT2":
        g_x = nx.read_gpickle("./word_embeddings/encoded_bert_v2.gpickle")    
    if word_embedding_encoding == "BERT_SIMPLE":
        g_x = nx.read_gpickle("./word_embeddings/encoded_bert_simple.gpickle")    

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


## Training
#### Evaluation methods
# Accuracy based on thresholds of distance (e.g. cosine > 0.8 should be a positive pair)
def threshold_acc(model, g, features, mask,loss,print_details=False,threshold_dist=0.2,threshold_cos=0.8,path=None):
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
    
    positives = 0.0
    negatives = 0.0
    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    for i in range(len(labels_tensor)):
        prediction = indices_tensor[i].item()
        label = labels_tensor[i].item()
        if label == 0.0 or label == -1.0:
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
    output['precision'] = true_positives / (true_positives + false_positives)
    output['fscore'] = 2 * (output['precision'] * output['recall']) / ((output['precision'] + output['recall']))
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

def confusion_matrix_cv(model, g, features, path,loss,threshold):
    model.eval()
    cv_number = path.split("tmp_cv_result_")[1].split(".")[0]
    path=path.split("/net_name")[0].replace("models/","datasets/")+"/"+cv_number
    tmp_test = pd.read_csv(path+"/test.csv").to_numpy()
    for mask in tmp_test:
        mask[0] = map_ds["DS_"+str(mask[0])]
        mask[1] = map_ds["DS_"+str(mask[1])]
        
    with th.no_grad():
        acc = threshold_acc(model, g, features, tmp_test,loss,print_details=True,threshold_dist=threshold,threshold_cos=threshold)
        return acc    
        
def evaluate(model, g, features, mask,loss):
    model.eval()
    with th.no_grad():
        #naive way of testing accuracy 
        th_output = threshold_acc(model, g, features, mask,loss)
        #accuracy based on 1-NN 
        if strategy == "isolation":
            acc2 = ne_ne_acc_isolation(model, g, features, mask,loss)
        if strategy == "random":
            acc2 = ne_ne_acc_random(model, g, features, mask,loss)
        return th_output,acc2

import time
import numpy as np
def train(training,iterations):
    dur = []
    print("Start of training...NN: "+training.net_name)
    #set max accuracy found if model already has state
    max_acc = 0.0
    max_acc2 = 0.0
    if len(training.log) > 0:
        for l in training.log:
            if l["fscore"] > max_acc:
                max_acc = l["fscore"]
                max_acc2 = l["acc2"]
            else:        
                if l["fscore"] == max_acc and l["acc2"] > max_acc2:
                    max_acc2 = l["acc2"]
            
    not_improving = 0
    
    #specify number of threads for the training
    #th.set_num_threads(2)
    
    for epoch in range(iterations):
        #model train mode
        training.net.train()
        t0 = time.time()
        epoch_loss = 0
        
        ## create batchs and shuffle data for training
        np.random.shuffle(train_mask)
        numb_splits = int(len(train_mask) / training.batch_splits) + 1
        train_batch = np.array_split(train_mask,numb_splits)
        
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
        th_output,acc2 = evaluate(training.net, g, g.ndata['vector'], test_mask,training.loss_name)
        
        #create log
        output = {}
        output['epoch'] = training.epochs_run
        output['loss'] = epoch_loss
        output['acc'] = th_output['acc']
        output['acc2'] = acc2
#         output['loss'] = float('%.5f'% (epoch_loss))
#         output['acc'] = float('%.5f'% (th_output['acc']))
#         output['acc2'] = float('%.5f'% (acc2))
        
        output['true_positives'] = th_output['true_positives']
        output['false_positives'] = th_output['false_positives']
        output['true_negatives'] = th_output['true_negatives']
        output['false_negatives'] = th_output['false_negatives']
        output['recall'] = th_output['recall']
        output['specificity'] = th_output['specificity']
        output['precision'] = th_output['precision']
        output['fscore'] = th_output['fscore']
        
        #updated parameters
        output['lr'] = training.lr
        output['batch_splits'] = training.batch_splits
        
        output['time_epoch'] = float('%.5f'% (np.mean(dur)))
        output['time_total'] = float('%.5f'% (training.runtime_seconds))
        training.log.append(output)
        training.epochs_run+=1
        print(str("Ep: {}, loss: {:.5f}, acc: {:.5f}, acc2: {:.5f}, prec: {:.5f}, rec: {:.5f}, fs: {:.5f}, time: {:.5f}, timeT: {:.5f}").format(output['epoch'],output['loss'],output['acc'],output['acc2'],output['precision'],output['recall'],output['fscore'],output['time_epoch'],output['time_total']))
        
        ##save best model and results found so far
        if output['fscore'] > max_acc:
            print("Best model found so far...")
            training.set_best(training)
            max_acc = output['fscore']
            max_acc2 = acc2
            not_improving = 0
        else:        
            if output['fscore'] == max_acc and acc2 > max_acc2:
                print("Best model found so far...")
                training.set_best(training)
                max_acc = output['fscore']
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
                                    
    #save final model state and final results if experiment is not a CV
    if cross_v < 0:
        if training.best != None:
            training.best.epochs_run = training.epochs_run
        training.save_state(path_setup)
        
        
def train2_deprecated(training,iterations):
    dur = []
    print("Start of training...NN: "+training.net_name)
    #set max accuracy found if model already has state
    max_acc = 0.0
    max_acc2 = 0.0
    if len(training.log) > 0:
        for l in training.log:
            if l["fscore"] > max_acc:
                max_acc = l["fscore"]
                max_acc2 = l["acc2"]
            else:        
                if l["fscore"] == max_acc and l["acc2"] > max_acc2:
                    max_acc2 = l["acc2"]
            
    not_improving = 0
    need_update = 0
    need_smaller_split = 0
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
        np.random.shuffle(train_mask)
        numb_splits = int(len(train_mask) / training.batch_splits) + 1
        train_batch = np.array_split(train_mask,numb_splits)
        
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
        th_output,acc2 = evaluate(training.net, g, g.ndata['vector'], test_mask,training.loss_name)
        
        #create log
        output = {}
        output['epoch'] = training.epochs_run
        output['loss'] = epoch_loss
        output['acc'] = th_output['acc']
        output['acc2'] = acc2
#         output['loss'] = float('%.5f'% (epoch_loss))
#         output['acc'] = float('%.5f'% (th_output['acc']))
#         output['acc2'] = float('%.5f'% (acc2))
        
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
        print(str("Ep: {}, loss: {:.5f}, fs: {:.5f}, rec: {:.5f}, prec: {:.5f},  time: {:.5f}, timeT: {:.5f}").format(output['epoch'],output['loss'],output['fscore'],output['recall'],output['precision'],output['time_epoch'],output['time_total']))
        
        ##save best model and results found so far
        if output['fscore'] > max_acc:
            print("##########Best model found so far##########")
            training.set_best(training)
            max_acc = output['fscore']
            max_acc2 = acc2
            not_improving = 0
            need_update = 0
            need_smaller_split = 0
        else:
            if need_update < 14:
                need_update += 1
            else:
                eex = str("{:e}".format(training.lr)).split("e")[1]
                decrease = "2e"+eex
                if float(str("{:e}".format(training.lr)).split("e")[0]) < 2.0:
                    d = float(decrease) / 10.0
                else:
                    d = float(decrease)
                new_lr = max([abs(training.lr - d),training.lr / 2.0])
                print(">>>>>>>>>>>{:.1e} decreased(-or/) by: {:.1e} gives {:.1e}".format(training.lr,d,new_lr))
                training.net = training.best.net
                training.optimizer = training.best.optimizer
                training.set_lr(max([new_lr,1e-3]))
                need_update = 0
    
            if need_smaller_split <44:
                need_smaller_split +=1
            else:
                training.net = training.best.net
                training.optimizer = training.best.optimizer
                training.batch_splits = max([training.batch_splits / 2, 32])
                need_smaller_split = 25
                print(">>>>>>>>>>>Updated batch size to: "+str(training.batch_splits))
            
            if not_improving < 135:
                not_improving +=1
            else:
                print("Not improving anymore...finishing training.")
                pad = iterations - epoch -1
                training.epochs_run+=pad
                break
                                    
    #Recover initial setup to save file
    training.lr = o_lr
    training.batch_splits = o_splits
    training.best.lr = o_lr 
    training.best.batch_splits = o_splits
    #save final model state and final results if experiment is not a CV
    if cross_v < 0:
        if training.best != None:
            training.best.epochs_run = training.epochs_run
        training.save_state(path_setup)
        

def train2(training,iterations):
    dur = []
    print("Start of training...NN: "+training.net_name)
    #set max accuracy found if model already has state
    max_acc = 0.0
    max_acc2 = 0.0
    if len(training.log) > 0:
        for l in training.log:
            if l["fscore"] > max_acc:
                max_acc = l["fscore"]
                max_acc2 = l["acc2"]
            else:        
                if l["fscore"] == max_acc and l["acc2"] > max_acc2:
                    max_acc2 = l["acc2"]
            
    not_improving = 0
    need_update = 0
    need_smaller_split = 0
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
        np.random.shuffle(train_mask)
        numb_splits = int(len(train_mask) / training.batch_splits) + 1
        train_batch = np.array_split(train_mask,numb_splits)
        
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
        th_output,acc2 = evaluate(training.net, g, g.ndata['vector'], test_mask,training.loss_name)
        
        #create log
        output = {}
        output['epoch'] = training.epochs_run
        output['loss'] = epoch_loss
        output['acc'] = th_output['acc']
        output['acc2'] = acc2
#         output['loss'] = float('%.5f'% (epoch_loss))
#         output['acc'] = float('%.5f'% (th_output['acc']))
#         output['acc2'] = float('%.5f'% (acc2))
        
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
        print(str("Ep: {}, loss: {:.5f}, fs: {:.5f}, rec: {:.5f}, prec: {:.5f},  time: {:.5f}, timeT: {:.5f}").format(output['epoch'],output['loss'],output['fscore'],output['recall'],output['precision'],output['time_epoch'],output['time_total']))
        
        ##save best model and results found so far
        if output['fscore'] > max_acc:
            print("##########Best model found so far##########")
            training.set_best(training)
            max_acc = output['fscore']
            max_acc2 = acc2
            not_improving = 0
            need_update = 0
            need_smaller_split = 0
        else:
            
            if not_improving < 100:
                not_improving +=1
            else:
                print("Not improving anymore...finishing training.")
                pad = iterations - epoch -1
                training.epochs_run+=pad
                break
            
            if need_smaller_split <39:
                need_smaller_split +=1
                if need_update < 19:
                    need_update += 1
                else:
                    eex = str("{:e}".format(training.lr)).split("e")[1]
                    decrease = "2e"+eex
                    if float(str("{:e}".format(training.lr)).split("e")[0]) < 2.0:
                        d = float(decrease) / 10.0
                    else:
                        d = float(decrease)
                    new_lr = max([abs(training.lr - d),training.lr / 2.0])
#                     training.net = training.best.net
#                     training.optimizer = training.best.optimizer
#                     training.set_best(training.best)
                    training.set_lr(max([new_lr,1e-3]))
                    need_update = 0
                    print(">>>>>>>>>>>Updated LR to {:.1e}. Batch size is  {}".format(training.lr,training.batch_splits))
            else:
#                 training.net = training.best.net
#                 training.optimizer = training.best.optimizer
#                 training.set_best(training.best)
                training.batch_splits = max([training.batch_splits / 2, 32])
                need_smaller_split = 0
                need_update = 0
                print(">>>>>>>>>>>Updated Batch size to {}. LR is  {:.1e}".format(training.batch_splits,training.lr))
                                    
    #Recover initial setup to save file
    training.lr = o_lr
    training.batch_splits = o_splits
    training.best.lr = o_lr 
    training.best.batch_splits = o_splits
    #save final model state and final results if experiment is not a CV
    if cross_v < 0:
        if training.best != None:
            training.best.epochs_run = training.epochs_run
        training.save_state(path_setup)                

def train3(training,iterations):
    dur = []
    print("Start of training...NN: "+training.net_name)
    #set max accuracy found if model already has state
    max_acc = 0.0
    max_acc2 = 0.0
    if len(training.log) > 0:
        for l in training.log:
            if l["fscore"] > max_acc:
                max_acc = l["fscore"]
                max_acc2 = l["acc2"]
            else:        
                if l["fscore"] == max_acc and l["acc2"] > max_acc2:
                    max_acc2 = l["acc2"]
            
    not_improving = 0
    need_update = 0
    need_smaller_split = 0
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
        np.random.shuffle(train_mask)
        numb_splits = int(len(train_mask) / training.batch_splits) + 1
        train_batch = np.array_split(train_mask,numb_splits)
        
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
        th_output,acc2 = evaluate(training.net, g, g.ndata['vector'], test_mask,training.loss_name)
        
        #create log
        output = {}
        output['epoch'] = training.epochs_run
        output['loss'] = epoch_loss
        output['acc'] = th_output['acc']
        output['acc2'] = acc2
#         output['loss'] = float('%.5f'% (epoch_loss))
#         output['acc'] = float('%.5f'% (th_output['acc']))
#         output['acc2'] = float('%.5f'% (acc2))
        
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
        print(str("Ep: {}, loss: {:.5f}, fs: {:.5f}, rec: {:.5f}, prec: {:.5f},  time: {:.5f}, timeT: {:.5f}").format(output['epoch'],output['loss'],output['fscore'],output['recall'],output['precision'],output['time_epoch'],output['time_total']))
        
        ##save best model and results found so far
        if output['fscore'] > max_acc:
            print("##########Best model found so far##########")
            training.set_best(training)
            max_acc = output['fscore']
            max_acc2 = acc2
            not_improving = 0
            need_update = 0
            need_smaller_split = 0
        else:
            
            if not_improving < 100:
                not_improving +=1
            else:
                print("Not improving anymore...finishing training.")
                pad = iterations - epoch -1
                training.epochs_run+=pad
                break
            
            
            if need_update < 19:
                need_update += 1
            else:
                eex = str("{:e}".format(training.lr)).split("e")[1]
                decrease = "2e"+eex
                if float(str("{:e}".format(training.lr)).split("e")[0]) < 2.0:
                    d = float(decrease) / 10.0
                else:
                    d = float(decrease)
                new_lr = max([abs(training.lr - d),training.lr / 2.0])
#                 training.net = training.best.net
#                 training.optimizer = training.best.optimizer
#                 training.set_best(training.best)
                training.set_lr(max([new_lr,1e-3]))
                need_update = 0
                print(">>>>>>>>>>>Updated LR to {:.1e}. Batch size is  {}".format(training.lr,training.batch_splits))
                                    
    #Recover initial setup to save file
    training.lr = o_lr
    training.batch_splits = o_splits
    training.best.lr = o_lr 
    training.best.batch_splits = o_splits
    #save final model state and final results if experiment is not a CV
    if cross_v < 0:
        if training.best != None:
            training.best.epochs_run = training.epochs_run
        training.save_state(path_setup)        
        
        

def train4(training,iterations):
    dur = []
    print("Start of training...NN: "+training.net_name)
    #set max accuracy found if model already has state
    max_acc = 0.0
    max_acc2 = 0.0
    if len(training.log) > 0:
        for l in training.log:
            if l["fscore"] > max_acc:
                max_acc = l["fscore"]
                max_acc2 = l["acc2"]
            else:        
                if l["fscore"] == max_acc and l["acc2"] > max_acc2:
                    max_acc2 = l["acc2"]
            
    not_improving = 0
    need_update = 0
    need_smaller_split = 0
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
        np.random.shuffle(train_mask)
        numb_splits = int(len(train_mask) / training.batch_splits) + 1
        train_batch = np.array_split(train_mask,numb_splits)
        
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
        th_output,acc2 = evaluate(training.net, g, g.ndata['vector'], test_mask,training.loss_name)
        
        #create log
        output = {}
        output['epoch'] = training.epochs_run
        output['loss'] = epoch_loss
        output['acc'] = th_output['acc']
        output['acc2'] = acc2
#         output['loss'] = float('%.5f'% (epoch_loss))
#         output['acc'] = float('%.5f'% (th_output['acc']))
#         output['acc2'] = float('%.5f'% (acc2))
        
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
        print(str("Ep: {}, loss: {:.5f}, fs: {:.5f}, rec: {:.5f}, prec: {:.5f},  time: {:.5f}, timeT: {:.5f}").format(output['epoch'],output['loss'],output['fscore'],output['recall'],output['precision'],output['time_epoch'],output['time_total']))
        
        ##save best model and results found so far
        if output['fscore'] > max_acc:
            print("##########Best model found so far##########")
            training.set_best(training)
            max_acc = output['fscore']
            max_acc2 = acc2
            not_improving = 0
#             training.set_lr(training.lr * .99)
        else:
            training.set_lr(training.lr * .99)
            if not_improving < 100:
                not_improving +=1
            else:
                print("Not improving anymore...finishing training.")
                pad = iterations - epoch -1
                training.epochs_run+=pad
                break
                                    
    #Recover initial setup to save file
    training.lr = o_lr
    training.batch_splits = o_splits
    training.best.lr = o_lr 
    training.best.batch_splits = o_splits
    #save final model state and final results if experiment is not a CV
    if cross_v < 0:
        if training.best != None:
            training.best.epochs_run = training.epochs_run
        training.save_state(path_setup)                
        
        
def train5(training,iterations):
    dur = []
    print("Start of training...NN: "+training.net_name)
    #set max accuracy found if model already has state
    max_acc = 0.0
    max_acc2 = 0.0
    if len(training.log) > 0:
        for l in training.log:
            if l["fscore"] > max_acc:
                max_acc = l["fscore"]
                max_acc2 = l["acc2"]
            else:        
                if l["fscore"] == max_acc and l["acc2"] > max_acc2:
                    max_acc2 = l["acc2"]
            
    not_improving = 0
    need_update = 0
    need_smaller_split = 0
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
        np.random.shuffle(train_mask)
        numb_splits = int(len(train_mask) / training.batch_splits) + 1
        train_batch = np.array_split(train_mask,numb_splits)
        
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
        th_output,acc2 = evaluate(training.net, g, g.ndata['vector'], test_mask,training.loss_name)
        
        #create log
        output = {}
        output['epoch'] = training.epochs_run
        output['loss'] = epoch_loss
        output['acc'] = th_output['acc']
        output['acc2'] = acc2
#         output['loss'] = float('%.5f'% (epoch_loss))
#         output['acc'] = float('%.5f'% (th_output['acc']))
#         output['acc2'] = float('%.5f'% (acc2))
        
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
        print(str("Ep: {}, loss: {:.5f}, fs: {:.5f}, rec: {:.5f}, prec: {:.5f},  time: {:.5f}, timeT: {:.5f}").format(output['epoch'],output['loss'],output['fscore'],output['recall'],output['precision'],output['time_epoch'],output['time_total']))
        
        ##save best model and results found so far
        if output['fscore'] > max_acc:
            print("##########Best model found so far##########")
            training.set_best(training)
            max_acc = output['fscore']
            max_acc2 = acc2
            not_improving = 0
            need_smaller_split = 0
        else:
            training.set_lr(training.lr * .99)
            if not_improving < 100:
                not_improving +=1
            else:
                print("Not improving anymore...finishing training.")
                pad = iterations - epoch -1
                training.epochs_run+=pad
                break
            
            if need_smaller_split <49:
                need_smaller_split +=1
                if need_smaller_split <17:
                    training.set_lr(training.lr * .85)
                    
            else:
                training.batch_splits = max([training.batch_splits / 2, 32])
                need_smaller_split = 0
                print(">>>>>>>>>>>Updated Batch size to {}".format(training.batch_splits))
                                    
    #Recover initial setup to save file
    training.lr = o_lr
    training.batch_splits = o_splits
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
        load_env(ds_name=dataset_name,ns=nsample,st=strategy,sp=create,we=word_embedding_encoding,cv=i)
        training_copy = copy.deepcopy(training)
        train5(training_copy,iterations)
        path_setup = dataset_name+"/"+strategy+"/"+str(neg_sample)+"/cv"
        if training_copy.best != None:
            training_copy.best.epochs_run = training_copy.epochs_run
        training_copy.save_state(path_setup,"/tmp_cv_result_"+str(i))
