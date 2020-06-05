from step3 import step3_gcn_nn_concatenate as gcn_nn
from step3 import step3_gcn_loss as gcn_loss
import torch as th
import os    
from pathlib import Path
import copy
# print(gcn_nn.get_options())
# print(gcn_loss.get_options())
# print(gcn_nn.get_instance(0,None))
# print(gcn_loss.get_instance(0,None,"0.5+sum"))

class Training():
    def __init__(self, net=None,batch_splits=None,lr=None,loss=None,loss_parameters=None,optimizer=None,optimizer_name=None):
        self.net = net
        self.net_name = type(net)
        self.batch_splits = batch_splits
        self.lr = lr
        self.loss = loss
        self.loss_name = type(loss)
        self.loss_parameters = loss_parameters
        self.optimizer_name = optimizer_name 
        self.optimizer = optimizer
        self.epochs_run = 0
        self.path = None
        self.runtime_seconds = 0
        self.log = []
        self.gen_path = ""
        self.best = None
        
    def set_training(self, net_name,batch_splits,lr,loss_name,loss_parameters,optimizer_name="adam"):
        self.net_name = net_name
        self.optimizer_name = optimizer_name
        self.batch_splits = batch_splits
        self.lr = lr
        self.loss_name = loss_name
        self.loss_parameters = loss_parameters
        
        self.loss = gcn_loss.get_instance(None,self.loss_name,parameters=self.loss_parameters)
        self.net = gcn_nn.get_instance(None,self.net_name)
        
        #initialize optimizer according user selection
        if self.optimizer_name == "adam":
            self.optimizer = th.optim.Adam(self.net.parameters(),self.lr,weight_decay=0.001)
        if self.optimizer_name == "sgd":
            self.optimizer = th.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.001)
            
    def set_best(self,best):
        best_copy = copy.deepcopy(best)
        best_copy.best = None
        self.best = best_copy
    
    def set_lr(self,lr):
        self.lr = lr
        if self.optimizer != None:
            self.optimizer.param_groups[0]['lr']=self.lr
        
        
    
    def save_state(self,path_setup="",cv_path=""):
        state = {}
        if self.net != None:
            state['net'] = self.net.state_dict()
            state['net_name'] = self.net_name
            state['batch_splits'] = self.batch_splits
            state['lr'] = self.lr
            state['loss_name'] = self.loss_name
            state['loss_parameters'] = self.loss_parameters
            state['optimizer_name'] = self.optimizer_name
            state['optimizer'] = self.optimizer.state_dict()
            state['epochs_run'] = self.log[-1]['epoch']+1
            state['log'] = self.log
            state['runtime_seconds'] = self.runtime_seconds
            from_epoch = "00"
            
            save_dir = str("net_name:{} / optimizer_name:{} / loss_name:{}"
                           .format(self.net_name,
                                   self.optimizer_name,
                                   self.loss_name
                           )).replace(" ","")
            
            if self.path!=None:
                parent_path = self.path.split("/")[-1] 
                from_epoch = parent_path.split(".pt")[0].split("|")[-1].split(":")[1].split("_")[-1]
                save_dir = save_dir+"/"+parent_path.split(".pt")[0]+"/"
            
            
            #standard e notation for lr
            enot = str("{:e}".format(self.lr))
            enot = enot.split("e")[0].rstrip('0').rstrip(".")+"e"+enot.split("e")[1]
            
            save_path = str("loss_parameters:{} | batch_splits:{} | lr:{} |  epochs_run:{}".
                            format(self.loss_parameters,
                                   self.batch_splits,
                                   enot,
                                   from_epoch+"_"+str("{:02d}".format(self.epochs_run))
                                  )).replace(" ","")
            
            outdir = path_setup+"/"+save_dir
            outpath = outdir +"/"+save_path
            self.gen_path = outpath
            
            if cv_path == "":
                outdir_model = "./models/"+ outdir
                if not os.path.exists(outdir_model):
                    Path(outdir_model).mkdir(parents=True, exist_ok=True)

                outdir_result = "./results/"+ outdir
                if not os.path.exists(outdir_result):
                    Path(outdir_result).mkdir(parents=True, exist_ok=True)
                
                path_model = outdir_model+"/"+save_path+".pt"
                path_result = outdir_result+"/"+save_path+".txt"
                
            else:
                outdir_model = "./models/"+ outdir +"/"+save_path
                if not os.path.exists(outdir_model):
                    Path(outdir_model).mkdir(parents=True, exist_ok=True)

                outdir_result = "./results/"+ outdir +"/"+save_path
                if not os.path.exists(outdir_result):
                    Path(outdir_result).mkdir(parents=True, exist_ok=True)
                
                path_model = outdir_model+cv_path+".pt"
                path_result = outdir_result+cv_path+".txt"
                
            
            th.save(state, path_model)
            file_out = open(path_result,'w') 
            file_out.writelines(str(self.log))
            file_out.close()
            print("Model and results saved")
            
            if self.best != None:
                print("Saving best model...")
                self.best.save_state(path_setup+"/best",cv_path)
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
        
        #retrocompatibility check
        if "optimizer_name" not in state.keys():
            if "betas" in state['optimizer']["param_groups"][0].keys():
                self.optimizer_name = "adam"
            else:
                self.optimizer_name = "sgd"
        else:
            self.optimizer_name = state['optimizer_name']
            
        #initialize optimizer according user selection
        if self.optimizer_name == "adam":
            self.optimizer = th.optim.Adam(self.net.parameters(),self.lr,weight_decay=0.001)
        if self.optimizer_name == "sgd":
            self.optimizer = th.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.001)
        
        #load states of NN and optimizer
        self.net.load_state_dict(state['net'])
        self.optimizer.load_state_dict(state['optimizer'])
        print("Training state loaded for configuration: \n" + path.split("/")[-1])
        print("Previous log: \n")
        print(self.log)    