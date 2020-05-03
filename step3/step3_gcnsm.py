#!/usr/bin/env python
# coding: utf-8

# # Get dataset with ~80% train, ~20% test

# In[ ]:


import numpy as np
import pandas as pd
import step3_dataset_setup as ds_setup
from step3 import step3_train_test_split as ds_split

def parameter_error(param_error):
    print("Encounter error in parameter setup "+param_error+", default values will be used: strategy=random, neg_sample=2 and create_new_split=True")
    neg_sample = 2
    strategy = "random"
    create_new_split = True
    word_embedding_encoding = "FASTTEXT"
neg_sample = ds_setup.neg_sample
strategy = ds_setup.strategy
create_new_split = ds_setup.create_new_split
word_embedding_encoding = ds_setup.word_embedding_encoding

print("Values to load")
print("neg_sample= "+str(neg_sample))
print("strategy= "+strategy)
print("create_new_split= "+str(create_new_split))
print("word_embedding_encoding= "+word_embedding_encoding)

#see dataset_setup to config parametrs for split data
if neg_sample == None or not int(neg_sample) or neg_sample < 0: 
    parameter_error("neg_sample")
if strategy == None or not str(strategy) or strategy not in ["isolation","random"]:
    parameter_error("strategy")
if create_new_split == None:
    parameter_error("create_new_split")
if word_embedding_encoding == None or not str(word_embedding_encoding) or word_embedding_encoding not in ["BERT","FASTTEXT"]:
    parameter_error("word_embedding_encoding")
    
if create_new_split:
    path_setup = ds_split.split_ds(strategy,neg_sample)
else:
    path_setup = strategy+"/"+str(neg_sample)

train_mask = pd.read_csv("./datasets/"+path_setup+"/train.csv").to_numpy()
test_mask = pd.read_csv("./datasets/"+path_setup+"/test.csv").to_numpy()

#info about split
train_positive = np.array([x for x in train_mask if x[2]==1])
test_positive = np.array([x for x in test_mask if x[2]==1])
print("Dataset splits loaded")
print("Train samples: "+str(len(train_mask)) + " Test samples: "+str(len(test_mask)))
print("Train positive samples: "+str(len(train_positive)) + " Test positive samples: "+str(len(test_positive)))


# # Read graph of metafeatures

# In[ ]:


import networkx as nx 

if word_embedding_encoding == "FASTTEXT":
    g_x = nx.read_gpickle("./word_embeddings/encoded_fasttext.gpickle")
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


# ### Export graph to deep graph library

# In[ ]:


import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
#convert from networkx to graph deep library format
g = dgl.DGLGraph()
g.from_networkx(g_x,node_attrs=['tipo','vector','ds_order'], edge_attrs=None)
g_x = None

print("Meta-feature graph from datasets loaded")


# # Training

# ### Evaluation methods

# In[ ]:


# Accuracy based on thresholds of distance (e.g. cosine > 0.8 should be a positive pair)
def threshold_acc(model, g, features, mask,loss):
    indices = []
    labels = []
    
    #mask = np.array([x for x in mask if x[2]==1])
    
    z1, z2 = model(g,features,mask[:,0],mask[:,1])
    
    #dist() | m - dist()
    if loss == "ContrastiveLoss" or loss == "Euclidean":
        pdist = th.nn.PairwiseDistance(p=2)        
        result = pdist(z1,z2)
        for i in range(len(result)):
            r = result[i]
            if r.item() <= 0.2:
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
            if r.item() >= 0.8:
                indices.append(1.0)
            else:
                indices.append(0.0)
        indices_tensor = th.tensor(indices)
        labels_tensor = th.tensor(mask[:,2])
    
    correct = th.sum(indices_tensor == labels_tensor)
    return correct.item() * 1.0 / len(labels_tensor)

# Accuracy based on nearest neighboor (e.g. the nearest node should be a positive pair)
def ne_ne_acc(model, g, features, mask,loss):

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


def evaluate(model, g, features, mask,loss):
    model.eval()
    with th.no_grad():
        #naive way of testing accuracy 
        acc = threshold_acc(model, g, features, mask,loss)
        #accuracy based on 1-NN 
        acc2 = ne_ne_acc(model, g, features, mask,loss)
        return acc,acc2


# In[ ]:


# evaluate(training.net,g,g.ndata['vector'],test_mask,training.loss_name)


# ### Train loop

# In[ ]:


import time
import numpy as np
def train(training,iterations):
    dur = []
    max_acc = 0.0
    ## training.splits indicates number of sets to split, not batch size!
    train_batch = np.array_split(train_mask,training.batch_splits)
    
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
        if acc2 + acc > max_acc:
            print("Best model found so far...")
            training.save_state(path_setup+"/best")
            max_acc = acc2 + acc 
        
    #save final model state and final results
    training.save_state(path_setup)


# ### Config and run training
# ### NN architectures: 
# {<br>
#     "0": "Bert_300", <br>
#     "1": "Bert_300_300_200", <br>
#     "2": "Bert_768", <br>
#     "3": "Fasttext3GCN_300" <br>
#     "4": "Fasttext_150", <br>
#     "5": "Fasttext_150_150_100", <br>
#     "6": "Fasttext_300" <br>
# }
# ### Loss functions: 
# {<br>
#     "0": "ContrastiveLoss", <br>
#     "1": "CosineEmbeddingLoss", <br>
# }
# ### Example to define architecture and loss
# <b>from step3 import step3_gcn_nn_concatenate as gcn_nn</b> <br>
# <b>from step3 import step3_gcn_loss as gcn_loss</b> <br>
# print(gcn_nn.get_options()) #list of options<br>
# print(gcn_loss.get_options()) #list of options<br>
# 
# ### Load training class to save/load/train experiments:
# <b>from step3 import step3_gcn_train as gcn_train</b>

# In[ ]:


from step3 import step3_gcn_nn_concatenate as gcn_nn
from step3 import step3_gcn_loss as gcn_loss
from step3 import step3_gcn_training as gcn_training
# #load model from path
# training = gcn_training.Training()
# training.load_state(path="./models/[file_name].pt")
# train(training,iterations=N)

# #train new model and specify parameters
# training = gcn_training.Training()
# training.set_training(
#             net_name= gcn_nn.get_option_name(),  #_of_option for NN architecture
#             batch_splits= ,#_of_sets(this will (give dataset / batch_splits) size of batch
#             lr= , #learning rate for training (e.g. 1e-3 )
#             loss_name=gcn_loss.get_option_name() #_of_option for loss ,
#             loss_parameters=) #loss function parameters separated by '+' e.g. for cosine and contrastive "0.0+mean" 
# train(training,iterations=N)


# ### Test suite

# In[ ]:


# training = gcn_training.Training()
# training.load_state(path="./models/net_name_Fasttext_300_batch_splits_28.0000_lr_0.0010_loss_name_ContrastiveLoss_loss_parameters_0.7+mean.pt")
# train(training,iterations=N)


# #Train with contrastive loss
# #train new model and specify parameters
# # training = gcn_training.Training()
# # training.set_training(
# #             net_name= gcn_nn.get_option_name(3),
# #             batch_splits=14,
# #             lr=1e-2,
# #             loss_name=gcn_loss.get_option_name(0),
# #             loss_parameters="1.0+mean")
# # train(training,iterations=40)

# # training = gcn_training.Training()
# # # training.load_state(path="./models/net_name:Fasttext_150|batch_splits:14.0000|lr:0.0010|loss_name:ContrastiveLoss|loss_parameters:0.5+mean.pt")
# # training.set_training(
# #             net_name= gcn_nn.get_option_name(3),
# #             batch_splits=14,
# #             lr=1e-2,
# #             loss_name=gcn_loss.get_option_name(1),
# #             loss_parameters="0.7+mean")
# # train(training,iterations=40)

# #train new model and specify parameters
# training = gcn_training.Training()
# training.set_training(
#             net_name= gcn_nn.get_option_name(4),
#             batch_splits=14,
#             lr=1e-3,
#             loss_name=gcn_loss.get_option_name(0),
#             loss_parameters="0.7+mean")
# train(training,iterations=10)


# In[ ]:




