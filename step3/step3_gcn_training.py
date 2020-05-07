from step3 import step3_gcn_nn_concatenate as gcn_nn
from step3 import step3_gcn_loss as gcn_loss
import torch as th
import os    
from pathlib import Path
# print(gcn_nn.get_options())
# print(gcn_loss.get_options())
# print(gcn_nn.get_instance(0,None))
# print(gcn_loss.get_instance(0,None,"0.5+sum"))

class Training():
    def __init__(self, net=None,batch_splits=None,lr=None,loss=None,loss_parameters=None,optimizer=None):
        self.net = net
        self.net_name = type(net)
        self.batch_splits = batch_splits
        self.lr = lr
        self.loss = loss
        self.loss_name = type(loss)
        self.loss_parameters = loss_parameters
        self.optimizer = optimizer
        self.epochs_run = 0
        self.path = None
        self.runtime_seconds = 0
        self.log = []
        
    def set_training(self, net_name,batch_splits,lr,loss_name,loss_parameters):
        self.net_name = net_name
        self.batch_splits = batch_splits
        self.lr = lr
        self.loss_name = loss_name
        self.loss_parameters = loss_parameters

        self.loss = gcn_loss.get_instance(None,self.loss_name,parameters=self.loss_parameters)
        self.net = gcn_nn.get_instance(None,self.net_name)
        self.optimizer = th.optim.Adam(self.net.parameters(),self.lr,weight_decay=0.001)
    
    def save_state(self,path_setup=""):
        state = {}
        if self.net != None:
            state['net'] = self.net.state_dict()
            state['net_name'] = self.net_name
            state['batch_splits'] = self.batch_splits
            state['lr'] = self.lr
            state['loss_name'] = self.loss_name
            state['loss_parameters'] = self.loss_parameters
            state['optimizer'] = self.optimizer.state_dict()
            state['epochs_run'] = self.epochs_run
            state['log'] = self.log
            state['runtime_seconds'] = self.runtime_seconds
            from_epoch = "00"
            if self.path!=None:
                from_epoch = self.path.split("/")[-1].split(".pt")[0].split("|")[-1].split(":")[1].split("_")[-1]
            
            save_path = str("net_name:{} | batch_splits:{} | lr:{:.4f} | \
            loss_name:{} | loss_parameters:{} | epochs_run:{}".
                            format(self.net_name,
                                   self.batch_splits,
                                   self.lr,
                                   self.loss_name,
                                   self.loss_parameters,
                                   from_epoch+"_"+str("{:02d}".format(self.epochs_run))
                                  )).replace(" ","")
            
            outdir = "./models/"+path_setup
            if not os.path.exists(outdir):
                Path(outdir).mkdir(parents=True, exist_ok=True)
            outdir = "./results/"+path_setup
            if not os.path.exists(outdir):
                Path(outdir).mkdir(parents=True, exist_ok=True)
                
            path_model = "./models/"+path_setup+"/"+save_path+".pt"
            path_result = "./results/"+path_setup+"/"+save_path+".txt"
            th.save(state, path_model)
            file_out = open(path_result,'w') 
            file_out.writelines(str(self.log))
            file_out.close()
            print("Model and results saved")
        else:
            print("Nothing to save")
        
    
    def load_state(self, path):
        state = th.load(path)
        self.net_name = state['net_name']
        self.batch_splits = state['batch_splits'] 
        self.lr = state['lr']
        self.loss_name = state['loss_name']
        self.loss_parameters = state['loss_parameters']
        self.epochs_run = state['epochs_run']
        self.log = state['log']
        self.path = path
        self.runtime_seconds = state['runtime_seconds']        
        self.loss = gcn_loss.get_instance(None,self.loss_name,parameters=self.loss_parameters)
        self.net = gcn_nn.get_instance(None,self.net_name)
        self.net.load_state_dict(state['net'])
        self.optimizer = th.optim.Adam(self.net.parameters(),self.lr,weight_decay=0.001)
        self.optimizer.load_state_dict(state['optimizer'])
        print("Training state loaded for configuration: \n" + path.split("/")[-1])
        print("Previous log: \n")
        print(self.log)    