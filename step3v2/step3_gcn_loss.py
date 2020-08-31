import torch as th
import sys, inspect

class ContrastiveLoss(th.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=0.5,reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, output1, output2, label):
        pdist = th.nn.PairwiseDistance(p=2)
        norm_euclidean = pdist(output1, output2) 
        
        if self.reduction == 'sum':
            loss_contrastive = th.sum( 0.5 * (1+label) * th.pow(norm_euclidean, 2) +
                                      0.5 * (1-label) * th.pow(th.clamp(self.margin -
                                                                        norm_euclidean, min=0.0), 2))
        if self.reduction == 'mean':
            loss_contrastive = th.mean( 0.5 * (1+label) * th.pow(norm_euclidean, 2) +
                                      0.5 * (1-label) * th.pow(th.clamp(self.margin - 
                                                                        norm_euclidean, min=0.0), 2))

        return loss_contrastive
    
class Euclidean(th.nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.pdist = th.nn.PairwiseDistance(p=2)

    def forward(self, output1, output2, label):
        loss_euclidean = th.mean( 0.5 * (1+label) * euclidean_distance +
                                      0.5 * (1-label) * th.clamp(- euclidean_distance, min=0.0))
        return loss_euclidean
    
class CosineEmbeddingLoss(th.nn.CosineEmbeddingLoss):
    def __init__(self, margin=0.0,reduction='mean'):
        super(CosineEmbeddingLoss, self).__init__(margin=margin, reduction=reduction)  
    
def get_options():
    list_nn = {}
    i = 0
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            nn = str(obj).split(".")[-1].split("'")[0]
            if nn != "[Some_Exception_in_class]":
                list_nn[i] = nn
                i+=1
    return list_nn

def get_option_name(option):
    return get_options()[option]

def get_instance(option=None,name=None,parameters=None):
    if option!=None:
        name = get_option_name(option)
    loss_class = getattr(sys.modules[__name__], name)
    
    if name == "Euclidean":
        return loss_class()
    
    if name == "CosineEmbeddingLoss" or name == "ContrastiveLoss":
        margin = None
        reduction = None
        if parameters != None:
            margin = float(parameters.split("+")[0])
            reduction = parameters.split("+")[1]     
        return loss_class(margin = margin, reduction = reduction)