## Get dataset with ~80% train, ~20% test
import numpy as np
import pandas as pd
import step3_train_test_split as ds_split
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
path_setup = None
dataset_name = "openml_203ds_datasets_matching"
cross_v=-1
g = None

def parameter_error(param_error,value):
    raise NameError("Encounter error in parameter {}".format(param_error))

def load_env(ds_name=None,ns=None,experiment=None,new_split=None,cv=0): 
    global dataset_name
    global neg_sample
    global strategy
    global create_new_split
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
        
    if experiment == None or not str(experiment) or experiment not in ["10_cv","random_subsam","hold_out"]:
        parameter_error("strategy",strategy)
    else:
        strategy = experiment
    if new_split == None:
        parameter_error("create_new_split",create_new_split)
    else:
        create_new_split = new_split

    print("Values to load")
    print("dataset_name="+dataset_name)
    print("neg_sample= "+str(neg_sample))
    print("strategy= "+strategy)
    print("create_new_split= "+str(create_new_split))
    print("cross_v= "+str(cross_v))    

    if strategy == "hold_out":
        path_setup = dataset_name+"/"+strategy
        train_mask = pd.read_csv("./ground_truth/"+path_setup+"/train.csv").to_numpy()
        test_mask = pd.read_csv("./ground_truth/"+path_setup+"/test.csv").to_numpy()
    
    else:
        if cross_v == 0 and create_new_split:
            print("Creating cross validation splits...")
            path_setup = ds_split.split_ds(dataset_name,strategy)
        else:
            path_setup = dataset_name+"/"+strategy
            
        train_mask = pd.read_csv("./ground_truth/"+path_setup+"/"+str(cross_v)+"/train.csv").to_numpy()
        test_mask = pd.read_csv("./ground_truth/"+path_setup+"/"+str(cross_v)+"/test.csv").to_numpy()
    
    #info about split
    train_positive = np.array([x for x in train_mask if x[2]==1])
    test_positive = np.array([x for x in test_mask if x[2]==1])
    print("Dataset splits loaded")
#     print("Train samples: "+str(len(train_mask)) + " Test samples: "+str(len(test_mask)))
    print("Train positive samples: "+str(len(train_positive)) + " Test positive samples: "+str(len(test_positive)))
    load_dgl()


## Read graph of metafeatures
import networkx as nx 
map_nodes = None
map_reverse = None
def load_graph():
    global map_nodes
    global map_reverse
    global train_mask
    global test_mask
    
    g_x = nx.read_gpickle("../step2/output/"+dataset_name+"/nodes_embeddings.gpickle")

    node_order = 0
    for x,n in sorted(g_x.nodes(data=True)):
        n['node_order']=node_order
        node_order+=1

    nodes = [x.strip() for (x,y) in g_x.nodes(data=True)]
    nodes_order = [y['node_order'] for x,y in g_x.nodes(data=True)]
    map_nodes = dict(zip(nodes,nodes_order))
    map_reverse = dict(zip(nodes_order,nodes))

    for mask in train_mask:
        mask[0] = map_nodes[str(mask[0]).strip().replace('\xa0','')]
        mask[1] = map_nodes[str(mask[1]).strip().replace('\xa0','')]            
        if mask[2] == 0:
            mask[2] = -1
            
    for mask in test_mask:
        mask[0] = map_nodes[str(mask[0]).strip().replace('\xa0','')]
        mask[1] = map_nodes[str(mask[1]).strip().replace('\xa0','')]
        if mask[2] == 0:
            mask[2] = -1
        
    
    train_mask = train_mask.astype(np.float) 
    test_mask = test_mask.astype(np.float) 
    
    ##get only same number of negative pairs for test to have 50/50 pos and neg
    test_pos = [x for x in test_mask if x[2]==1.0]
    test_neg = [x for x in test_mask if x[2]==-1.0]
    np.random.shuffle(test_neg)
    test_mask = np.concatenate((test_pos,test_neg[0:len(test_pos)]))
    np.random.shuffle(test_mask)
    
    return g_x


## Export graph to deep graph library
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
#convert from networkx to graph deep library format
def load_dgl():
    global g
    g_x = load_graph()
    g = dgl.from_networkx(g_x,node_attrs=['vector','node_order'], edge_attrs=None)
#     g = dgl.DGLGraph()
#     g.from_networkx(g_x,node_attrs=['tipo','vector','node_order'], edge_attrs=None)
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

# def confusion_matrix(model, g, features, mask,loss,threshold):
def confusion_matrix(training,path=None):
    model = training.net
    loss = training.loss_name
    if strategy == "hold_out":
        model.eval()
        test_path = "./ground_truth/"+path_setup+"/test.csv"
        tmp_test = pd.read_csv(test_path).to_numpy()
        for mask in tmp_test:
            mask[0] = map_nodes[str(mask[0]).strip().replace('\xa0','')]
            mask[1] = map_nodes[str(mask[1]).strip().replace('\xa0','')]
            if mask[2] == 0:
                mask[2] = -1
        tmp_test = tmp_test.astype(np.float) 
        ##get only same number of negative pairs for test to have 50/50 pos and neg
        test_pos = [x for x in tmp_test if x[2]==1.0]
        test_neg = [x for x in tmp_test if x[2]==-1.0]
        np.random.shuffle(test_neg)
        tmp_test = np.concatenate((test_pos,test_neg[0:len(test_pos)]))
        np.random.shuffle(tmp_test)
        with th.no_grad():
            acc = threshold_acc(model, g, g.ndata['vector'], tmp_test,loss,print_details=True)
            return acc
    else:
        if strategy == "10_cv" or strategy == "random_subsam":   
            model.eval()
            cv_number = path.split("tmp_cv_result_")[1].split(".")[0]
            test_path = "./ground_truth/"+path_setup+"/"+cv_number+"/test.csv"
            tmp_test = pd.read_csv(test_path).to_numpy()
            for mask in tmp_test:
                mask[0] = map_nodes[str(mask[0]).strip().replace('\xa0','')]
                mask[1] = map_nodes[str(mask[1]).strip().replace('\xa0','')]
                if mask[2] == 0:
                    mask[2] = -1
            tmp_test = tmp_test.astype(np.float) 
            with th.no_grad():
                acc = threshold_acc(model, g, g.ndata['vector'], tmp_test,loss,print_details=True)
                return acc    
        else:
            raise NameError("Experiment: {} does not exists".format(strategy))
    
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
        
def train(training,iterations,nsample=2):
    dur = []
    print(str("Start of training...NN {} Loss {} Split {}: ").format(training.net_name,training.loss_name,training.batch_splits))
    #set max accuracy found if model already has state
    max_fscore = 0.0
    if len(training.log) > 0:
        for l in training.log:
            if l["fscore"] > max_fscore:
                max_fscore = l["fscore"]
            
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
        if output['fscore'] > max_fscore:
            print("##########Best model found so far##########")
            training.set_best(training)
            max_fscore = output['fscore']
        training.set_lr(training.lr * .99)       
        
                                    
    #Recover initial setup to save file
    training.lr = o_lr
    training.batch_splits = o_splits
    if training.best != None:
        training.best.lr = o_lr 
        training.best.batch_splits = o_splits
    
    #save final model state and final results if experiment is not a CV
    if strategy == "hold_out":
        if training.best != None:
            training.best.epochs_run = training.epochs_run
        training.save_state(path_setup)                        
        
def cross_validation(training,iterations=1,ran="1-10",nsample=None):
    global cv_logs
    
    if nsample == None:
        nsample = neg_sample
    
    cv_ran = ran.split("-")
    init = int(cv_ran[0]) -1
    ending = int(cv_ran[1])
    if init < 0 or (ending >100 and strategy=="isolation") or (ending >10 and strategy=="random"):
        raise Exception("Values for CV out of range")
    
    
    training_copy = None
    for i in range(init,ending):
        load_env(ds_name=dataset_name,ns=0,experiment=strategy,new_split=False,cv=i)
        training_copy = copy.deepcopy(training)
        train(training_copy,iterations,nsample)
        path_setup = dataset_name+"/"+strategy+"/"+str(nsample)
        if training_copy.best != None:
            training_copy.best.epochs_run = training_copy.epochs_run
        training_copy.save_state(path_setup,"/tmp_cv_result_"+str(i))
