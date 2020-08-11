#from dgl.nn.pytorch import GraphConv
import numpy as np
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import sys, inspect

gcn_msg = fn.copy_u(u='vector', out='m')
gcn_reduce = fn.sum(msg='m', out='vector')
gcn_reduce_mean = fn.mean(msg='m', out='vector')

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

class GCNLayer_sum(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer_sum, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g,feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['vector'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['vector'] + feature
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

class Fasttext_3GCN(nn.Module):
    def __init__(self):
        super(Fasttext_3GCN, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(600, 300)
        self.bn1 = nn.BatchNorm1d(num_features=300)
        self.layer2 = GCNLayer_concatenate(600, 300)
        self.bn2 = nn.BatchNorm1d(num_features=300)
        self.layer3 = GCNLayer_concatenate(600, 300)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        gcn = F.leaky_relu(self.layer3(g, gcn))
        
        z1 = F.normalize(gcn[v1], p=2, dim=1)
        
        z2 = F.normalize(gcn[v2], p=2, dim=1)
        
        return z1,z2
    
    
class Fasttext_simple_300_300(nn.Module):
    def __init__(self):
        super(Fasttext_simple_300_300, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(600, 300)
        self.bn1 = nn.BatchNorm1d(num_features=300)
        
        #particular layers
        self.layer2 = nn.Linear(300, 300)
        self.bn2 = nn.BatchNorm1d(num_features=300)
        self.layer3 = nn.Linear(300, 300)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        
        z1 = F.leaky_relu(self.layer2(gcn[v1]))
        z1 = F.leaky_relu(self.layer3(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer2(gcn[v2]))
        z2 = F.leaky_relu(self.layer3(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2
    
    
class Fasttext3GCN_300(nn.Module):
    def __init__(self):
        super(Fasttext3GCN_300, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(600, 300)
        self.bn1 = nn.BatchNorm1d(num_features=300)
        self.layer2 = GCNLayer_concatenate(600, 300)
        self.bn2 = nn.BatchNorm1d(num_features=300)
        self.layer3 = GCNLayer_concatenate(600, 300)
        self.bn2 = nn.BatchNorm1d(num_features=300)
        
        #particular layers
        self.layer4 = nn.Linear(300, 300)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        gcn = F.leaky_relu(self.layer3(g, gcn))
        
        z1 = F.leaky_relu(self.layer4(gcn[v1]))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer4(gcn[v2]))
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
    
class Fasttext_200_200_200_100(nn.Module):
    def __init__(self):
        super(Fasttext_200_200_200_100, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(600, 300)
        self.bn1 = nn.BatchNorm1d(num_features=300)
        self.layer2 = GCNLayer_concatenate(600, 300)
        self.bn2 = nn.BatchNorm1d(num_features=300)
        
        #particular layers
        self.layer3 = nn.Linear(300, 200)
        self.bn3 = nn.BatchNorm1d(num_features=200)
        self.layer4 = nn.Linear(200, 200)
        self.bn4 = nn.BatchNorm1d(num_features=200)
        self.layer5 = nn.Linear(200, 200)
        self.bn5 = nn.BatchNorm1d(num_features=200)
        self.layer6 = nn.Linear(200, 100)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.leaky_relu(self.layer5(z1))
        z1 = F.leaky_relu(self.layer6(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.leaky_relu(self.layer5(z2))
        z2 = F.leaky_relu(self.layer6(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2

class Fasttext_300_200_100(nn.Module):
    def __init__(self):
        super(Fasttext_300_200_100, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(600, 300)
        self.bn1 = nn.BatchNorm1d(num_features=300)
        self.layer2 = GCNLayer_concatenate(600, 300)
        self.bn2 = nn.BatchNorm1d(num_features=300)
        
        #particular layers
        self.layer3 = nn.Linear(300, 300)
        self.bn3 = nn.BatchNorm1d(num_features=300)
        self.layer4 = nn.Linear(300, 200)
        self.bn4 = nn.BatchNorm1d(num_features=200)
        self.layer5 = nn.Linear(200, 100)
        
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

    
class Fasttext2(nn.Module):
    def __init__(self):
        super(Fasttext2, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.normalize(gcn[v1], p=2, dim=1)
        z2 = F.normalize(gcn[v2], p=2, dim=1)
        
        return z1,z2    

    
class Fasttext2_150(nn.Module):
    def __init__(self):
        super(Fasttext2_150, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 150)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2    
    
class FasttextSum_150(nn.Module):
    def __init__(self):
        super(FasttextSum_150, self).__init__()
        
        self.layer1 = GCNLayer_sum(364, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_sum(364, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 150)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2
    
class FasttextSum_364(nn.Module):
    def __init__(self):
        super(FasttextSum_364, self).__init__()
        
        self.layer1 = GCNLayer_sum(364, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)        

        self.layer2 = GCNLayer_sum(364, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 364)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2    

    

class FasttextSum_364_300_200_100(nn.Module):
    def __init__(self):
        super(FasttextSum_364_300_200_100, self).__init__()
        
        self.layer1 = GCNLayer_sum(364, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_sum(364, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 364)
        self.bn3 = nn.BatchNorm1d(num_features=364)
        self.layer4 = nn.Linear(364, 300)
        self.bn4 = nn.BatchNorm1d(num_features=300)
        self.layer5 = nn.Linear(300, 200)
        self.bn5 = nn.BatchNorm1d(num_features=200)
        self.layer6 = nn.Linear(200, 100)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.leaky_relu(self.layer5(z1))
        z1 = F.leaky_relu(self.layer6(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.leaky_relu(self.layer5(z2))
        z2 = F.leaky_relu(self.layer6(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2    
    
class FasttextSum_364_200_100(nn.Module):
    def __init__(self):
        super(FasttextSum_364_200_100, self).__init__()
        
        self.layer1 = GCNLayer_sum(364, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_sum(364, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 364)
        self.bn3 = nn.BatchNorm1d(num_features=364)
        self.layer4 = nn.Linear(364, 200)
        self.bn4 = nn.BatchNorm1d(num_features=200)
        self.layer5 = nn.Linear(200, 100)
        
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
    
class FasttextSumd_150(nn.Module):
    def __init__(self):
        super(FasttextSumd_150, self).__init__()
        
        self.layer1 = GCNLayer_sum(364, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.dp1 = nn.Dropout()
        self.layer2 = GCNLayer_sum(364, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        self.dp2 = nn.Dropout()
        
        #particular layers
        self.layer3 = nn.Linear(364, 150)
        
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2
    
    
class Fasttext2_364(nn.Module):
    def __init__(self):
        super(Fasttext2_364, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 364)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2

class Fasttext2_728_364(nn.Module):
    def __init__(self):
        super(Fasttext2_728_364, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 728)
        self.bn2 = nn.BatchNorm1d(num_features=728)
        
        #particular layers
        self.layer3 = nn.Linear(728, 728)
        self.bn3 = nn.BatchNorm1d(num_features=728)
        self.layer4 = nn.Linear(728, 364)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2

class Fasttext_600_300(nn.Module):
    def __init__(self):
        super(Fasttext_600_300, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(600, 300)
        self.bn1 = nn.BatchNorm1d(num_features=300)
        self.layer2 = GCNLayer_concatenate(600, 600)
        self.bn2 = nn.BatchNorm1d(num_features=600)
        
        #particular layers
        self.layer3 = nn.Linear(600, 600)
        self.bn3 = nn.BatchNorm1d(num_features=600)
        self.layer4 = nn.Linear(600, 300)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2
    
    
class Fasttext2d_728_364(nn.Module):
    def __init__(self):
        super(Fasttext2d_728_364, self).__init__()
        self.dp1 = nn.Dropout(p=0.2)
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.dp2 = nn.Dropout()
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 728)
        self.dp3 = nn.Dropout()
        self.bn2 = nn.BatchNorm1d(num_features=728)
        
        #particular layers
        self.layer3 = nn.Linear(728, 728)
        self.dp4 = nn.Dropout()
        self.bn3 = nn.BatchNorm1d(num_features=728)
        self.layer4 = nn.Linear(728, 364)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2

    
class Fasttext2_364_nn(nn.Module):
    def __init__(self):
        super(Fasttext2_364_nn, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 364)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        
        return z1,z2
    
    
class Fasttext2_728(nn.Module):
    def __init__(self):
        super(Fasttext2_728, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 728)
        self.bn2 = nn.BatchNorm1d(num_features=728)
        
        #particular layers
        self.layer3 = nn.Linear(728, 728)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2
    
class Fasttext2_728364(nn.Module):
    def __init__(self):
        super(Fasttext2_728364, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 728)
        self.bn2 = nn.BatchNorm1d(num_features=728)
        
        #particular layers
        self.layer3 = nn.Linear(728, 364)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2
    
    
class Fasttext2_364_364_364(nn.Module):
    def __init__(self):
        super(Fasttext2_364_364_364, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 364)
        self.bn3 = nn.BatchNorm1d(num_features=364)
        self.layer4 = nn.Linear(364, 364)
        self.bn4 = nn.BatchNorm1d(num_features=364)
        self.layer5 = nn.Linear(364, 364)
        
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

class Fasttext_300_300_300(nn.Module):
    def __init__(self):
        super(Fasttext_300_300_300, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(600, 300)
        self.bn1 = nn.BatchNorm1d(num_features=300)
        self.layer2 = GCNLayer_concatenate(600, 300)
        self.bn2 = nn.BatchNorm1d(num_features=300)
        
        #particular layers
        self.layer3 = nn.Linear(300, 300)
        self.bn3 = nn.BatchNorm1d(num_features=300)
        self.layer4 = nn.Linear(300, 300)
        self.bn4 = nn.BatchNorm1d(num_features=300)
        self.layer5 = nn.Linear(300, 300)
        
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
    
class Fasttext2d_364_364_364(nn.Module):
    def __init__(self):
        super(Fasttext2d_364_364_364, self).__init__()
        self.dp1 = nn.Dropout(p=0.2)
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.dp2 = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.dp3 = nn.Dropout(p=0.5)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 364)
        self.dp4 = nn.Dropout(p=0.5)
        self.bn3 = nn.BatchNorm1d(num_features=364)
        self.layer4 = nn.Linear(364, 364)
        self.dp5 = nn.Dropout(p=0.5)
        self.bn4 = nn.BatchNorm1d(num_features=364)
        self.layer5 = nn.Linear(364, 364)
        
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

    
    
class Fasttext2d_364(nn.Module):
    def __init__(self):
        super(Fasttext2d_364, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.dp1 = nn.Dropout()
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        self.dp2 = nn.Dropout()
        
        #particular layers
        self.layer3 = nn.Linear(364, 364)
        
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

class Fasttext2_200_200(nn.Module):
    def __init__(self):
        super(Fasttext2_200_200, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 200)
        self.bn3 = nn.BatchNorm1d(num_features=200)
        self.layer4 = nn.Linear(200, 200)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2
    
class Fasttext2_200_200_100(nn.Module):
    def __init__(self):
        super(Fasttext2_200_200_100, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 200)
        self.bn3 = nn.BatchNorm1d(num_features=200)
        self.layer4 = nn.Linear(200, 200)
        self.bn4 = nn.BatchNorm1d(num_features=200)
        self.layer5 = nn.Linear(200, 100)
        
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

class Fasttext2_200_200_200(nn.Module):
    def __init__(self):
        super(Fasttext2_200_200_200, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 200)
        self.bn3 = nn.BatchNorm1d(num_features=200)
        self.layer4 = nn.Linear(200, 200)
        self.bn4 = nn.BatchNorm1d(num_features=200)
        self.layer5 = nn.Linear(200, 200)
        
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

class Fasttext2_364_200_100(nn.Module):
    def __init__(self):
        super(Fasttext2_364_200_100, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 364)
        self.bn3 = nn.BatchNorm1d(num_features=364)
        self.layer4 = nn.Linear(364, 200)
        self.bn4 = nn.BatchNorm1d(num_features=200)
        self.layer5 = nn.Linear(200, 100)
        
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

class Fasttext2_364_200_100_relu(nn.Module):
    def __init__(self):
        super(Fasttext2_364_200_100_relu, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 364)
        self.bn3 = nn.BatchNorm1d(num_features=364)
        self.layer4 = nn.Linear(364, 200)
        self.bn4 = nn.BatchNorm1d(num_features=200)
        self.layer5 = nn.Linear(200, 100)
        
    def forward(self, g,features,v1,v2):
        gcn = F.relu(self.layer1(g,features))
        gcn = F.relu(self.layer2(g, gcn))
        
        z1 = F.relu(self.layer3(gcn[v1]))
        z1 = F.relu(self.layer4(z1))
        z1 = F.relu(self.layer5(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.relu(self.layer3(gcn[v2]))
        z2 = F.relu(self.layer4(z2))
        z2 = F.relu(self.layer5(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2

    
    
class Fasttext2_300_250_200_150(nn.Module):
    def __init__(self):
        super(Fasttext2_300_250_200_150, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 300)
        self.bn3 = nn.BatchNorm1d(num_features=300)
        self.layer4 = nn.Linear(300, 250)
        self.bn4 = nn.BatchNorm1d(num_features=250)
        self.layer5 = nn.Linear(250, 200)
        self.bn5 = nn.BatchNorm1d(num_features=200)
        self.layer6 = nn.Linear(200, 150)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.leaky_relu(self.layer5(z1))
        z1 = F.leaky_relu(self.layer6(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.leaky_relu(self.layer5(z2))
        z2 = F.leaky_relu(self.layer6(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2    

class Fasttext2d_300_250_200_150(nn.Module):
    def __init__(self):
        super(Fasttext2d_300_250_200_150, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.dp1 = nn.Dropout()
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        self.dp2 = nn.Dropout()
        
        #particular layers
        self.layer3 = nn.Linear(364, 300)
        self.bn3 = nn.BatchNorm1d(num_features=300)
        self.dp3 = nn.Dropout()
        self.layer4 = nn.Linear(300, 250)
        self.bn4 = nn.BatchNorm1d(num_features=250)
        self.dp4 = nn.Dropout()
        self.layer5 = nn.Linear(250, 200)
        self.bn5 = nn.BatchNorm1d(num_features=200)
        self.dp5 = nn.Dropout()
        self.layer6 = nn.Linear(200, 150)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.leaky_relu(self.layer5(z1))
        z1 = F.leaky_relu(self.layer6(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.leaky_relu(self.layer5(z2))
        z2 = F.leaky_relu(self.layer6(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2    
    
    
class FasttextSum_300_250_200_150(nn.Module):
    def __init__(self):
        super(FasttextSum_300_250_200_150, self).__init__()
        
        self.layer1 = GCNLayer_sum(364, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_sum(364, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 300)
        self.bn3 = nn.BatchNorm1d(num_features=300)
        self.layer4 = nn.Linear(300, 250)
        self.bn4 = nn.BatchNorm1d(num_features=250)
        self.layer5 = nn.Linear(250, 200)
        self.bn5 = nn.BatchNorm1d(num_features=200)
        self.layer6 = nn.Linear(200, 150)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.leaky_relu(self.layer5(z1))
        z1 = F.leaky_relu(self.layer6(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.leaky_relu(self.layer5(z2))
        z2 = F.leaky_relu(self.layer6(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2    

class FasttextSumd_300_250_200_150(nn.Module):
    def __init__(self):
        super(FasttextSumd_300_250_200_150, self).__init__()
        
        self.layer1 = GCNLayer_sum(364, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.dp1 = nn.Dropout()
        self.layer2 = GCNLayer_sum(364, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        self.dp2 = nn.Dropout()
        
        #particular layers
        self.layer3 = nn.Linear(364, 300)
        self.bn3 = nn.BatchNorm1d(num_features=300)
        self.dp3 = nn.Dropout()
        self.layer4 = nn.Linear(300, 250)
        self.bn4 = nn.BatchNorm1d(num_features=250)
        self.dp4 = nn.Dropout()
        self.layer5 = nn.Linear(250, 200)
        self.bn5 = nn.BatchNorm1d(num_features=200)
        self.dp5 = nn.Dropout()
        self.layer6 = nn.Linear(200, 150)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.leaky_relu(self.layer5(z1))
        z1 = F.leaky_relu(self.layer6(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.leaky_relu(self.layer5(z2))
        z2 = F.leaky_relu(self.layer6(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2    
    
    
class Fasttext2_3GCN_300_250_200_150(nn.Module):
    def __init__(self):
        super(Fasttext2_3GCN_300_250_200_150, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        self.layer3 = GCNLayer_concatenate(728, 364)
        self.bn3 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer4 = nn.Linear(364, 300)
        self.bn4 = nn.BatchNorm1d(num_features=300)
        self.layer5 = nn.Linear(300, 250)
        self.bn5 = nn.BatchNorm1d(num_features=250)
        self.layer6 = nn.Linear(250, 200)
        self.bn6 = nn.BatchNorm1d(num_features=200)
        self.layer7 = nn.Linear(200, 150)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        gcn = F.leaky_relu(self.layer3(g, gcn))
        
        z1 = F.leaky_relu(self.layer4(gcn[v1]))
        z1 = F.leaky_relu(self.layer5(z1))
        z1 = F.leaky_relu(self.layer6(z1))
        z1 = F.leaky_relu(self.layer7(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer4(gcn[v2]))
        z2 = F.leaky_relu(self.layer5(z2))
        z2 = F.leaky_relu(self.layer6(z2))
        z2 = F.leaky_relu(self.layer7(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2    
    
    
class Fasttext_300_250_200_150(nn.Module):
    def __init__(self):
        super(Fasttext_300_250_200_150, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(600, 300)
        self.bn1 = nn.BatchNorm1d(num_features=300)
        self.layer2 = GCNLayer_concatenate(600, 300)
        self.bn2 = nn.BatchNorm1d(num_features=300)
        
        #particular layers
        self.layer3 = nn.Linear(300, 300)
        self.bn3 = nn.BatchNorm1d(num_features=300)
        self.layer4 = nn.Linear(300, 250)
        self.bn4 = nn.BatchNorm1d(num_features=250)
        self.layer5 = nn.Linear(250, 200)
        self.bn5 = nn.BatchNorm1d(num_features=200)
        self.layer6 = nn.Linear(200, 150)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.leaky_relu(self.layer5(z1))
        z1 = F.leaky_relu(self.layer6(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.leaky_relu(self.layer5(z2))
        z2 = F.leaky_relu(self.layer6(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2    

    
class Fasttext2_200_200_200_100(nn.Module):
    def __init__(self):
        super(Fasttext2_200_200_200_100, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 200)
        self.bn3 = nn.BatchNorm1d(num_features=200)
        self.layer4 = nn.Linear(200, 200)
        self.bn4 = nn.BatchNorm1d(num_features=200)
        self.layer5 = nn.Linear(200, 200)
        self.bn5 = nn.BatchNorm1d(num_features=200)
        self.layer6 = nn.Linear(200, 100)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.leaky_relu(self.layer5(z1))
        z1 = F.leaky_relu(self.layer6(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.leaky_relu(self.layer5(z2))
        z2 = F.leaky_relu(self.layer6(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2

class Fasttext2_200_200_200_100_relu(nn.Module):
    def __init__(self):
        super(Fasttext2_200_200_200_100_relu, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 200)
        self.bn3 = nn.BatchNorm1d(num_features=200)
        self.layer4 = nn.Linear(200, 200)
        self.bn4 = nn.BatchNorm1d(num_features=200)
        self.layer5 = nn.Linear(200, 200)
        self.bn5 = nn.BatchNorm1d(num_features=200)
        self.layer6 = nn.Linear(200, 100)
        
    def forward(self, g,features,v1,v2):
        gcn = F.relu(self.layer1(g,features))
        gcn = F.relu(self.layer2(g, gcn))
        
        z1 = F.relu(self.layer3(gcn[v1]))
        z1 = F.relu(self.layer4(z1))
        z1 = F.relu(self.layer5(z1))
        z1 = F.relu(self.layer6(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.relu(self.layer3(gcn[v2]))
        z2 = F.relu(self.layer4(z2))
        z2 = F.relu(self.layer5(z2))
        z2 = F.relu(self.layer6(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2


class Fasttext2_364_300_200_100(nn.Module):
    def __init__(self):
        super(Fasttext2_364_300_200_100, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 364)
        self.bn3 = nn.BatchNorm1d(num_features=364)
        self.layer4 = nn.Linear(364, 300)
        self.bn4 = nn.BatchNorm1d(num_features=300)
        self.layer5 = nn.Linear(300, 200)
        self.bn5 = nn.BatchNorm1d(num_features=200)
        self.layer6 = nn.Linear(200, 100)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.leaky_relu(self.layer5(z1))
        z1 = F.leaky_relu(self.layer6(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.leaky_relu(self.layer5(z2))
        z2 = F.leaky_relu(self.layer6(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2

    
class Fasttext2_364_300_250_200(nn.Module):
    def __init__(self):
        super(Fasttext2_364_300_250_200, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 364)
        self.bn3 = nn.BatchNorm1d(num_features=364)
        self.layer4 = nn.Linear(364, 300)
        self.bn4 = nn.BatchNorm1d(num_features=300)
        self.layer5 = nn.Linear(300, 250)
        self.bn5 = nn.BatchNorm1d(num_features=250)
        self.layer6 = nn.Linear(250, 200)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.leaky_relu(self.layer5(z1))
        z1 = F.leaky_relu(self.layer6(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.leaky_relu(self.layer5(z2))
        z2 = F.leaky_relu(self.layer6(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2    
class Fasttext2_364_300_200_100_relu(nn.Module):
    def __init__(self):
        super(Fasttext2_364_300_200_100_relu, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(728, 364)
        self.bn1 = nn.BatchNorm1d(num_features=364)
        self.layer2 = GCNLayer_concatenate(728, 364)
        self.bn2 = nn.BatchNorm1d(num_features=364)
        
        #particular layers
        self.layer3 = nn.Linear(364, 364)
        self.bn3 = nn.BatchNorm1d(num_features=364)
        self.layer4 = nn.Linear(364, 300)
        self.bn4 = nn.BatchNorm1d(num_features=300)
        self.layer5 = nn.Linear(300, 200)
        self.bn5 = nn.BatchNorm1d(num_features=200)
        self.layer6 = nn.Linear(200, 100)
        
    def forward(self, g,features,v1,v2):
        gcn = F.relu(self.layer1(g,features))
        gcn = F.relu(self.layer2(g, gcn))
        
        z1 = F.relu(self.layer3(gcn[v1]))
        z1 = F.relu(self.layer4(z1))
        z1 = F.relu(self.layer5(z1))
        z1 = F.relu(self.layer6(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.relu(self.layer3(gcn[v2]))
        z2 = F.relu(self.layer4(z2))
        z2 = F.relu(self.layer5(z2))
        z2 = F.relu(self.layer6(z2))
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

class Bert2_400(nn.Module):
    def __init__(self):
        super(Bert2_400, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(1664, 832)
        self.bn1 = nn.BatchNorm1d(num_features=768)
        self.layer2 = GCNLayer_concatenate(1664, 832)
        self.bn2 = nn.BatchNorm1d(num_features=832)
        
        #particular layers
        self.layer3 = nn.Linear(832, 400)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2

    
class Bert2_832_400_200_100(nn.Module):
    def __init__(self):
        super(Bert2_832_400_200_100, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(1664, 832)
        self.bn1 = nn.BatchNorm1d(num_features=768)
        self.layer2 = GCNLayer_concatenate(1664, 832)
        self.bn2 = nn.BatchNorm1d(num_features=832)
        
        #particular layers
        self.layer3 = nn.Linear(832, 832)
        self.bn3 = nn.BatchNorm1d(num_features=832)
        self.layer4 = nn.Linear(832, 400)
        self.bn4 = nn.BatchNorm1d(num_features=400)
        self.layer5 = nn.Linear(400, 200)
        self.bn5 = nn.BatchNorm1d(num_features=200)
        self.layer6 = nn.Linear(200, 100)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.leaky_relu(self.layer5(z1))
        z1 = F.leaky_relu(self.layer6(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.leaky_relu(self.layer5(z2))
        z2 = F.leaky_relu(self.layer6(z2))
        z2 = F.normalize(z2, p=2, dim=1)
        
        return z1,z2
    
class Bert2_832_600_400_200(nn.Module):
    def __init__(self):
        super(Bert2_832_600_400_200, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(1664, 832)
        self.bn1 = nn.BatchNorm1d(num_features=768)
        self.layer2 = GCNLayer_concatenate(1664, 832)
        self.bn2 = nn.BatchNorm1d(num_features=832)
        
        #particular layers
        self.layer3 = nn.Linear(832, 832)
        self.bn3 = nn.BatchNorm1d(num_features=832)
        self.layer4 = nn.Linear(832, 600)
        self.bn4 = nn.BatchNorm1d(num_features=600)
        self.layer5 = nn.Linear(600, 400)
        self.bn5 = nn.BatchNorm1d(num_features=400)
        self.layer6 = nn.Linear(400, 200)
        
    def forward(self, g,features,v1,v2):
        gcn = F.leaky_relu(self.layer1(g,features))
        gcn = F.leaky_relu(self.layer2(g, gcn))
        
        z1 = F.leaky_relu(self.layer3(gcn[v1]))
        z1 = F.leaky_relu(self.layer4(z1))
        z1 = F.leaky_relu(self.layer5(z1))
        z1 = F.leaky_relu(self.layer6(z1))
        z1 = F.normalize(z1, p=2, dim=1)
        
        z2 = F.leaky_relu(self.layer3(gcn[v2]))
        z2 = F.leaky_relu(self.layer4(z2))
        z2 = F.leaky_relu(self.layer5(z2))
        z2 = F.leaky_relu(self.layer6(z2))
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

class Bert2_832(nn.Module):
    def __init__(self):
        super(Bert2_832, self).__init__()
        
        self.layer1 = GCNLayer_concatenate(1664, 832)
        self.bn1 = nn.BatchNorm1d(num_features=832)
        self.layer2 = GCNLayer_concatenate(1664, 832)
        self.bn2 = nn.BatchNorm1d(num_features=832)
        
        #particular layers
        self.layer3 = nn.Linear(832, 832)
        
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
            if nn not in ["GCNLayer_concatenate","GCNLayer_sum"]:
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
        