#!/usr/bin/env python
# coding: utf-8

# # Read dataset and plot graph

# In[1]:


import numpy as np
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
from random import randrange

def read_dataset(path,drop_columns=None,keep_columns=None):
    #get rid of useless columns
    csv_data = pd.read_csv(path,sep="~")
    
    if keep_columns != None:
        #keep only these columns
        return csv_data.filter(items=keep_columns)
    
    if drop_columns!= None:
        #drop these and keep the rest
        return csv_data.drop(drop_columns, axis=1)
    
    #finally, didn't drop or filter any column
    return csv_data     

def plot_graph(g,ds_nodes=[],attribute_nodes=[],feat_nodes=[],lit_nodes=[]):
    pos=nx.spring_layout(g)    
    nx.draw_networkx_nodes(g,pos,nodelist=ds_nodes,node_color="blue",node_size=900)
    nx.draw_networkx_nodes(g,pos,nodelist=attribute_nodes,node_color="green",node_size=900)
    nx.draw_networkx_nodes(g,pos,nodelist=feat_nodes,node_color="grey",node_size=900)
    nx.draw_networkx_nodes(g,pos,nodelist=lit_nodes,node_color="red",node_size=900)

    nx.draw_networkx_edges(g,pos,width=3)
    nx.draw_networkx_labels(g,pos,font_size=8)
    plt.show() 


# ## Graph  construction

# In[2]:


def code_id(data,parent):
    return parent+"//"+data


# In[3]:


def graph_dataset_short(datasets,g=None,wem="fasttext",instances=0):
    if g == None:
        g = nx.Graph()
    
    #create nodes and edges at datasetLevel
    features = datasets.columns[2:]
    
    if instances==0:
        number_instances = len(datasets)
    else:
        number_instances = instances
    
    for r in range(number_instances): 
        #node id is the openML id which is in the first column
        dataset_id = str(datasets.iloc[r][0])
        g.add_node(dataset_id,vector=word_embedding("dataset|"+datasets.iloc[r][1] ,wem),tipo="dataset")
        row = datasets.iloc[r][2:]
        
        if instances == 0:
            number_features = len(features)
        else:
            number_features = min(instances,len(features))
            
        for i in range (number_features):
            feature_dataset_id = code_id(features[i],dataset_id)
            g.add_node(feature_dataset_id,vector=word_embedding(features[i]+"|"+str(row[i]) ,wem),tipo="feature_dataset")
            g.add_edge(dataset_id,feature_dataset_id)
            
    return g


def graph_attribute_short(datasets,g=None,wem="fasttext",instances=0):
    if g == None:
        g = nx.Graph()
        
    #create nodes and edges at datasetLevel
    features = datasets.columns[2:]
    
    if instances==0:
        number_instances = len(datasets)
    else:
        number_instances = min (instances,len(datasets))
    
    for r in range(number_instances): 
        #node id is the openML id which is in the first column
        #attr name is the 2nd column
        dataset_id = str(datasets.iloc[r][0])
        attribute_id = code_id(str(datasets.iloc[r][1]),dataset_id)
        row = datasets.iloc[r][2:]
        
        g.add_node(attribute_id,vector=word_embedding("attribute|"+datasets.iloc[r][1],wem),tipo="attribute")
        
        #relation of dataset and an attribute
        g.add_edge(dataset_id,attribute_id)
        
        if instances == 0:
            number_features = len(features)
        else:
            number_features = min (instances,len(features))
            
        for i in range (number_features):
            feature_attribute_id = code_id(str(features[i]),attribute_id)
            g.add_node(feature_attribute_id,vector=word_embedding(features[i]+"|"+str(row[i]),wem),tipo="feature_attribute")
            g.add_edge(attribute_id,feature_attribute_id)
    return g


# ## Auxiliars

# ### Check if input is number

# In[4]:


def is_number(s):
    #Returns True is string is a number.
    try:
        float(s)
        if float(s) == float("INF") or float(s) == float("NAN") or s == "NAN" or s == "nan":
            return False
        return True
    except ValueError:
        return False


# ### From numbers to bin tensor vector

# In[46]:


from decimal import Decimal
import bitstring
import torch
#clean
def num2vec(num):
    rep_sc = str('{:.11E}'.format(num))
#     print(rep_sc)
    dec_part = int(rep_sc.split("E")[0].replace(".",""))
    c = 1
    if dec_part <0:
        c = -1
    dec_part = abs(dec_part)
    
    exp_part = int(rep_sc.split("E")[1])
    if exp_part <0:
        exp_pos = 0
        exp_neg = exp_part
    else:
        exp_pos = exp_part
        exp_neg = 0

    exp_pos = abs(exp_pos)    
    exp_neg = abs(exp_neg)
    
    rep_str = str("{:03}{:03}{:012}".format(exp_pos,exp_neg,dec_part))
#     print(rep_str)
    
#     print(dec_part)
    rep_int = int(rep_str) * c
    rep_bin = bitstring.Bits(int=rep_int, length=64).bin
#     print(rep_bin)

    bin_tensor = torch.tensor(np.array([float(x) for x in rep_bin]))
    return bin_tensor


# ## Fasttext

# In[6]:


import numpy as np
import torch
import fasttext
#fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('./resources/fasttext.bin')
print(ft.get_dimension())


# In[7]:


def fasttex_simple(value):
    if is_number(value):
        value = str(value)
    
    values = value.split("|")
    out_tensor = torch.zeros(300)
    for v in values:
        out_tensor = out_tensor + torch.tensor(ft.get_sentence_vector(value))
    out_tensor = out_tensor / len(values)
    return out_tensor
    
def fasttex_(value):
    value = str(value)
    values = value.split("|")
    out_tensor = torch.zeros(364)
    for v in values:
        if is_number(v):
            value_f = float(v)
            bin_tensor = num2vec(value_f)
            out_tensor = out_tensor + torch.cat((torch.zeros(300),bin_tensor.float()))
        else:
            str_tensor = torch.tensor(ft.get_sentence_vector(value))
            out_tensor = out_tensor + torch.cat((str_tensor.float(),torch.zeros(64)))
    out_tensor = out_tensor / len(values)
    return out_tensor


# ## Choose word embedding

# In[6]:


def word_embedding(data, model):
    if model=="fasttext":
        return fasttex_(data)
    if model=="bert":
        return bert(data)
    if model=="fasttext_simple":
        return fasttex_simple(data)
    if model=="bert_simple":
        return bert_simple(data)


# # Execute

# In[10]:


#build graph
def step2(dataset):
    word_emb = "fasttext"
    g = g = nx.Graph()
    input_path = "../step1/output/"+dataset+"/"
    output_path = "./output/"+dataset+"/"
    df_dataset = read_dataset(input_path/"ds.csv",sep="~");
    g = graph_dataset_short(df_dataset,g,word_emb)
    df_attributes = read_dataset(input_path/"attr_nom.csv",sep="~");
    g = graph_attribute_short(df_attributes,g,word_emb)
    df_attributes_numeric = read_dataset(input_path/"attr_num.csv",sep="~");
    g = graph_attribute_short(df_attributes_numeric,g,word_emb)
    #write graph to file
    if not os.path.exists(output_path):
        Path(output_path).mkdir(parents=True, exist_ok=True)
    nx.write_gpickle(g, output_path+"nodes_embeddings.gpickle")
    print("node embeddings for "+dataset+" created")


# ## Read previously created graph

# In[7]:


#read
# g = nx.read_gpickle("../word_embeddings/clean_monitor_fasttext_short.gpickle")

