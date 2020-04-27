#from dgl.nn.pytorch import GraphConv
import numpy as np
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import sys, inspect

gcn_msg = fn.copy_src(src='vector', out='m')
gcn_reduce = fn.sum(msg='m', out='vector')

class GCNLayer_concatenate(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer_concatenate, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g,feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['vector'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = th.cat([feature,g.ndata['vector']],dim=1)
            g.ndata['vector'] = self.linear(h)
            return g.ndata['vector']


class Fasttext_300(nn.Module):
    def __init__(self):
        super(Fasttext_300, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(600, 300)
        self.bn1 = nn.BatchNorm1d(num_features=300)
        self.layer2 = GCNLayer_concatenate(600, 300)
        self.bn2 = nn.BatchNorm1d(num_features=300)
        
        #particular layers
        self.layer3 = nn.Linear(300, 300)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2
        
        
class Fasttext_150(nn.Module):
    def __init__(self):
        super(Fasttext_150, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(600, 300)
        self.bn1 = nn.BatchNorm1d(num_features=300)
        self.layer2 = GCNLayer_concatenate(600, 300)
        self.bn2 = nn.BatchNorm1d(num_features=300)
        
        #particular layers
        self.layer3 = nn.Linear(300, 150)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2
    
class Fasttext_150_150_100(nn.Module):
    def __init__(self):
        super(Fasttext_150_150_100, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(600, 300)
        self.bn1 = nn.BatchNorm1d(num_features=300)
        self.layer2 = GCNLayer_concatenate(600, 300)
        self.bn2 = nn.BatchNorm1d(num_features=300)
        
        #particular layers
        self.layer3 = nn.Linear(300, 150)
        self.bn3 = nn.BatchNorm1d(num_features=150)
        self.layer4 = nn.Linear(150, 150)
        self.bn4 = nn.BatchNorm1d(num_features=150)
        self.layer5 = nn.Linear(150, 100)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.leaky_relu(self.layer5(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.leaky_relu(self.layer5(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2

# class Fasttext_180_60_6_2(nn.Module):
#     def __init__(self):
#         super(Fasttext_180_60_6_2, self).__init__()
#         self.layer1 = GCNLayer(600, 300)
#         self.layer2 = GCNLayer(600, 300)
#         self.layer3 = nn.Linear(300, 180)
        
#         self.layer4 = nn.Linear(180, 60)
        
#         self.layer_softmax = nn.Linear(60, 6)
#         self.layer_logistic = nn.Linear(60, 2)
    
#     def forward(self, g,features):
#         x = F.leaky_relu(self.layer1(g,features))
#         x = self.layer2(g, x)
#         x = th.tanh(self.layer3(x))
#         x = self.layer4(features)
#         return x
    
#     def forward_softmax(self, features):
#         x = th.tanh(self.layer_softmax(x))
#         return x
    
#     def forward_logistic(self, features):
#         x = th.tanh(self.layer_logistic(x))
#         return x    
        
class Bert_300(nn.Module):
    def __init__(self):
        super(Bert_300, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(1536, 768)
        self.bn1 = nn.BatchNorm1d(num_features=768)
        self.layer2 = GCNLayer_concatenate(1536, 768)
        self.bn2 = nn.BatchNorm1d(num_features=768)
        
        #particular layers
        self.layer3 = nn.Linear(768, 300)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2
    
    
class Bert_768(nn.Module):
    def __init__(self):
        super(Bert_768, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(1536, 768)
        self.bn1 = nn.BatchNorm1d(num_features=768)
        self.layer2 = GCNLayer_concatenate(1536, 768)
        self.bn2 = nn.BatchNorm1d(num_features=768)
        
        #particular layers
        self.layer3 = nn.Linear(768, 768)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2

class Bert_300_300_200(nn.Module):
    def __init__(self):
        super(Bert_300_300_200, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(1536, 768)
        self.bn1 = nn.BatchNorm1d(num_features=768)
        self.layer2 = GCNLayer_concatenate(1536, 768)
        self.bn2 = nn.BatchNorm1d(num_features=768)
        
        #particular layers
        self.layer3 = nn.Linear(768, 300)
        self.bn3 = nn.BatchNorm1d(num_features=300)
        self.layer4 = nn.Linear(300, 300)
        self.bn4 = nn.BatchNorm1d(num_features=300)
        self.layer5 = nn.Linear(300, 200)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.leaky_relu(self.layer5(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.leaky_relu(self.layer5(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2
    
def get_options():
    list_nn = {}
    i = 0
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            nn = str(obj).split(".")[-1].split("'")[0]
            if nn != "GCNLayer_concatenate":
                list_nn[i] = nn
                i+=1
    return list_nn

def get_option_name(option):
    return get_options()[option]

def get_instance(option=None,name=None):
    if option!=None:
        name = get_option_name(option)
    gcn_class = getattr(sys.modules[__name__], name)
    gcn_instance = gcn_class()
    return gcn_instance
        