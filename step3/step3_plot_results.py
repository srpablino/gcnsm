#!/usr/bin/env python
# coding: utf-8

# # Plot results

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def reading(file_path):
    s = open(file_path, 'r').read()
    return eval(s)

#parameters for iterations
colors = ["red","green","blue","orange","black","pink","grey","skyblue","yellow","brown"]
loss_parameters = ["0.9+mean","0.7+mean","0.5+mean","0.4+mean","0.3+mean","0.35+mean","0.1+mean"]
learning_rates = [1e-2,2e-2,4e-2,5e-2,1e-3,2e-3,4e-3,5e-3,8e-3,5e-4,7e-4]
splits = ["32","64","128","256","512","1024"]
ep_run_init = ["00","30","50","60","100","125","150","175","200","250","300"]
ep_run_end = ["50","60","70","80","100","125","150","175","200","250","300","400"]
# ep_run_end = ["100","125","150","175","200","250","300","400"]

#options
db_name = ["openml_203ds_datasets_matching"]
strategy = ["isolation","random"]
archi = ["Fasttext_150","Fasttext_300","Bert_300","Bert_768"]
optimizer = ["adam","sgd"]
loss_functions = ["ContrastiveLoss","CosineEmbeddingLoss"]


# ## Details by loss parameters

# In[ ]:


def plot_by_loss_parameters(sampling,db,st,a,op,lf,subpath=""):
    ##option selection
#     sampling = 2
#     db = db_name[0]
#     st = strategy[0]
#     a = archi[0]
#     op = optimizer[1]
#     lf = loss_functions[0]

    path = str("./results/{}/{}/{}/net_name:{}/optimizer_name:{}/loss_name:{}/".format(db,st,sampling,a,op,lf))
    path += subpath
    for s in splits:
        for lr in learning_rates:
            for eri in ep_run_init:
                for ere in ep_run_end:
                    content = False
                    for lp in range(len(loss_parameters)):                    
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{:.0e}|epochs_run:{}_{}".format(loss_parameters[lp],s,lr,eri,ere))
                        
                        ##sub_folders
                        if os.path.exists(path+file_name+"/"):
                            plot_by_loss_parameters(sampling,db,st,a,op,lf,file_name+"/")
                        
                        try:
                            df_results0 = reading(path+file_name+".txt")
                        except:
    #                         print(path+file_name+".txt")
                            continue
                        content = True
#                         break

                    if not content:
                        continue

                    fig1, axs = plt.subplots(1, 2, figsize=(35, 10), facecolor='w', edgecolor='k')
                    for lp in range(len(loss_parameters)):                    
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{:.0e}|epochs_run:{}_{}".format(loss_parameters[lp],s,lr,eri,ere))
                        #{'epoch': 0, 'loss': 0.07563, 'acc': 0.54412, 'acc2': 0.36527, 'time_epoch': 268.2493, 'time_total': 268.2493}    
                        try:
                            df_results0 = reading(path+file_name+".txt")
                        except:
                            continue
                        loss = (np.array([x["loss"] for x in df_results0])*1000).tolist()
#                         loss = list(x["loss"] for x in df_results0)
                        acc = list(x["acc"] for x in df_results0)
                        acc2 = list(x["acc2"] for x in df_results0)

                        fig1.suptitle(str("NegSample: {}x Optimizer: {} Architecture: {} Loss: {} lr: {:.0e} splits: {}").format(sampling,op,a,lf,lr,s),fontsize=18)
                        
                        axs[0].yaxis.grid(True)
                        axs[0].tick_params(labelsize=16)

                        axs[1].set_ylim([0.5,1.0])
                        axs[1].set_yticks(np.arange(0.5, 1.05, 0.025))
                        axs[1].yaxis.grid(True)
                        axs[1].tick_params(labelsize=16)
                        
                        axs[0].plot(range(0,len(df_results0)), loss, marker="o", c=colors[lp],  linestyle='--', label=" Loss parameters: margin="+loss_parameters[lp])
                        axs[1].plot(range(0,len(df_results0)), acc, marker="o", c=colors[lp],  linestyle='--', label=" Loss parameters: margin="+loss_parameters[lp] + "max acc: "+ str(get_max_acc(acc)))

                        axs[0].set_title("Train loss (x1000)",fontsize=16)
                        axs[1].set_title("Test accuracy Trheshold",fontsize=16)
                        

                        leg = axs[0].legend(loc='best', ncol=1, shadow=True, fancybox=True,fontsize=16)
                        leg = axs[1].legend(loc='best', ncol=1, shadow=True, fancybox=True,fontsize=16)
                        
#                         axs[2].set_ylim([0,1])
#                         axs[2].set_yticks(np.arange(0, 1.1, 0.1))
#                         axs[2].yaxis.grid(True)
#                         axs[2].tick_params(labelsize=16)
#                         axs[2].plot(range(0,len(df_results0)), acc2, marker="o", c=colors[lp],  linestyle='--', label=" Loss parameters: margin="+loss_parameters[lp])
#                         axs[2].set_title("Test accuracy Nearest neighbor",fontsize=16)
#                         leg = axs[2].legend(loc='best', ncol=1, shadow=True, fancybox=True)
                            

                    fig1.savefig(path+str("batch_splits:{}|lr:{:.0e}|epochs_run:{}_{}".format(s,lr,eri,ere))+".png",pad_inches = 0)
# plot_by_loss_parameters()


# ## Details by split

# In[ ]:


def plot_by_split(sampling,db,st,a,op,lf,subpath=""):
    ##options selection
#     sampling = 2
#     db = db_name[0]
#     st = strategy[0]
#     a = archi[0]
#     op = optimizer[0]
#     lf = loss_functions[0]

    path = str("./results/{}/{}/{}/net_name:{}/optimizer_name:{}/loss_name:{}/".format(db,st,sampling,a,op,lf))
    path +=subpath
    for lp in range(len(loss_parameters)):                    
        for lr in learning_rates:
            for eri in ep_run_init:
                for ere in ep_run_end:
                    content = False
                    for s in range(len(splits)):
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{:.0e}|epochs_run:{}_{}".format(loss_parameters[lp],splits[s],lr,eri,ere))
                        
                        ##sub_folders
                        if os.path.exists(path+file_name+"/"):
                            plot_by_split(sampling,db,st,a,op,lf,file_name+"/")
                        
                        try:
                            df_results0 = reading(path+file_name+".txt")
                        except:
    #                         print(path+file_name+".txt")
                            continue
                        content = True
#                         break

                    if not content:
                        continue

                    fig1, axs = plt.subplots(1, 2, figsize=(35, 10), facecolor='w', edgecolor='k')
                    for s in range(len(splits)):
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{:.0e}|epochs_run:{}_{}".format(loss_parameters[lp],splits[s],lr,eri,ere))
                        #{'epoch': 0, 'loss': 0.07563, 'acc': 0.54412, 'acc2': 0.36527, 'time_epoch': 268.2493, 'time_total': 268.2493}    
                        try:
                            df_results0 = reading(path+file_name+".txt")
                        except:
                            continue
                        loss = (np.array([x["loss"] for x in df_results0])*1000).tolist()
#                         loss = list(x["loss"] for x in df_results0)
                        acc = list(x["acc"] for x in df_results0)
                        acc2 = list(x["acc2"] for x in df_results0)

                        fig1.suptitle(str("NegSample: {}x Optimizer: {} Architecture: {} Loss: {} lr: {:.0e} splits: {}").format(sampling,op,a,lf,lr,s),fontsize=18)
                        axs[1].set_ylim([0.5,1.0])
    #                     axs[2].set_ylim([0,1])

                        axs[1].set_yticks(np.arange(0.5, 1.05, 0.05))
                        axs[1].yaxis.grid(True)
                        axs[1].tick_params(labelsize=16)
                        axs[0].tick_params(labelsize=16)
                    
                        axs[0].plot(range(0,len(df_results0)), loss, marker="o", c=colors[s],  linestyle='--', label="split= "+splits[s])
                        axs[1].plot(range(0,len(df_results0)), acc, marker="o", c=colors[s],  linestyle='--', label="split= "+splits[s] + "max acc: "+ str(get_max_acc(acc)))
    #                     axs[2].plot(range(0,len(df_results0)), acc2, marker="o", c=colors[s],  linestyle='--', label="split= "+splits[s])

                        axs[0].set_title("Train loss (x1000)",fontsize=16)
                        axs[1].set_title("Test accuracy Trheshold",fontsize=16)
    #                     axs[2].set_title("Test accuracy Nearest neighbor")

                        leg = axs[0].legend(loc='best', ncol=1, shadow=True, fancybox=True,fontsize=16)
                        leg = axs[1].legend(loc='best', ncol=1, shadow=True, fancybox=True,fontsize=16)
    #                     leg = axs[2].legend(loc='best', ncol=1, shadow=True, fancybox=True)

                    fig1.savefig(path+str("loss_parameters:{}|lr:{:.0e}|epochs_run:{}_{}".format(loss_parameters[lp],lr,eri,ere))+".png",pad_inches = 0)
# plot_by_split()


# ## Cross Validation

# In[ ]:


def get_max_acc(accs):
    max_acc = 0.0
    for acc in accs:
        if acc > max_acc:
            max_acc = acc
    return max_acc
    
def plot_cv(sampling,db,st,a,op,lf,subpath=""):
    ##options selection
#     sampling = 2
#     db = db_name[0]
#     st = strategy[1]
#     a = archi[0]
#     op = optimizer[0]
#     lf = loss_functions[0]
    path = str("./results/{}/{}/{}/cv/net_name:{}/optimizer_name:{}/loss_name:{}/".format(db,st,sampling,a,op,lf))
    path += subpath
    for s in splits:
        for lr in learning_rates:
            for eri in ep_run_init:
                for ere in ep_run_end:
                    content = False
                    for lp in range(len(loss_parameters)):                    
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{:.0e}|epochs_run:{}_{}".format(loss_parameters[lp],s,lr,eri,ere))
                        
                        ##sub_folders
                        if os.path.exists(path+file_name+"/"):
                            plot_cv(sampling,db,st,a,op,lf,file_name+"/")
                        
                        try:
                            df_results0 = reading(path+file_name+"/tmp_cv_result.txt")
                        except:
    #                         print(path+file_name+".txt")
                            continue
                        content = True
#                         break

                    if not content:
                        continue

                    fig1, axs = plt.subplots(1, 2, figsize=(35, 10), facecolor='w', edgecolor='k')
                    for lp in range(len(loss_parameters)):                    
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{:.0e}|epochs_run:{}_{}".format(loss_parameters[lp],s,lr,eri,ere))
                        #{'epoch': 0, 'loss': 0.07563, 'acc': 0.54412, 'acc2': 0.36527, 'time_epoch': 268.2493, 'time_total': 268.2493}    
                        try:
                            df_results0 = reading(path+file_name+"/tmp_cv_result.txt")
                        except:
                            continue
                        max_acc_list = []
                        for i in  range (10):
                            logs = df_results0[i]
                            loss = (np.array([log["loss"] for log in logs])*1000).tolist()
#                             loss = [log["loss"] for log in logs]
                            acc = [log["acc"] for log in logs]
#                             acc2 = [log["acc2"] for log in logs]
                            max_acc = get_max_acc(acc)
                            max_acc_list.append(max_acc)
                            axs[1].set_ylim([0.5,1.0])
        #                     axs[2].set_ylim([0,1])

                            axs[1].set_yticks(np.arange(0.5, 1.05, 0.05))
                            axs[1].yaxis.grid(True)
                            axs[1].tick_params(labelsize=16)
                            axs[0].tick_params(labelsize=16)

                            axs[0].plot(range(0,len(logs)), loss, marker="o", c=colors[i],  linestyle='--', label="Run="+str(i+1))
                            axs[1].plot(range(0,len(logs)), acc, marker="o", c=colors[i],  linestyle='--', label="Run="+str(i+1)+" max_acc= "+str(max_acc))
        #                     axs[2].plot(range(0,len(df_results0)), acc2, marker="o", c=colors[lp],  linestyle='--', label=" Loss parameters: margin="+loss_parameters[lp])

                            axs[0].set_title("Train loss (x1000)",fontsize=16)
                            axs[1].set_title("Test accuracy Trheshold",fontsize=16)
        #                     axs[2].set_title("Test accuracy Nearest neighbor")

                            leg = axs[0].legend(loc='best', ncol=1, shadow=True, fancybox=True)
                            leg = axs[1].legend(loc='best', ncol=1, shadow=True, fancybox=True)
        #                     leg = axs[2].legend(loc='best', ncol=1, shadow=True, fancybox=True)
                        max_acc_avg = np.average(np.array(max_acc_list))
                        fig1.suptitle(file_name+"_Cross_validation. MAX ACC AVG = "+str(max_acc_avg),fontsize=18)
                        fig1.savefig(path+file_name+"/tmp_cv_results.png",pad_inches = 0)
                    
# plot_cv()

def plot_cv_2(sampling,db,st,a,op,lf,subpath=""):
    ##options selection
#     sampling = 2
#     db = db_name[0]
#     st = strategy[1]
#     a = archi[0]
#     op = optimizer[0]
#     lf = loss_functions[0]
    path = str("./results/{}/{}/{}/cv/net_name:{}/optimizer_name:{}/loss_name:{}/".format(db,st,sampling,a,op,lf))
    path += subpath
    for s in splits:
        for lr in learning_rates:
            for eri in ep_run_init:
                for ere in ep_run_end:
                    for lp in range(len(loss_parameters)): 
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{:.0e}|epochs_run:{}_{}".format(loss_parameters[lp],s,lr,eri,ere))

                        ##sub_folders
                        if os.path.exists(path+file_name+"/"):
                            plot_cv_2(sampling,db,st,a,op,lf,file_name+"/")
                                
                        content = False
                        for cv_i in range(10):
                            try:
                                df_ = reading(path+file_name+"/tmp_cv_result_"+str(cv_i)+".txt")
                            except:
        #                         print(path+file_name+".txt")
                                continue
                            content = True
    #                         break

                        if not content:
                            continue

                        fig1, axs = plt.subplots(1, 2, figsize=(35, 10), facecolor='w', edgecolor='k')
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{:.0e}|epochs_run:{}_{}".format(loss_parameters[lp],s,lr,eri,ere))
                        
                        df_results0 = []
                        for cv_i in range(10):                            
                            try:
                                df_ = reading(path+file_name+"/tmp_cv_result_"+str(cv_i)+".txt")
                            except:
                                continue
                            df_results0.append(df_[0])
                        max_acc_list = []
                        for i in  range (len(df_results0)):
                            logs = df_results0[i]
                            loss = (np.array([log["loss"] for log in logs])*1000).tolist()
#                             loss = [log["loss"] for log in logs]
                            acc = [log["acc"] for log in logs]
#                             acc2 = [log["acc2"] for log in logs]
                            max_acc = get_max_acc(acc)
                            max_acc_list.append(max_acc)
                            axs[1].set_ylim([0.5,1.0])
        #                     axs[2].set_ylim([0,1])

                            axs[1].set_yticks(np.arange(0.5, 1.05, 0.05))
                            axs[1].yaxis.grid(True)
                            axs[1].tick_params(labelsize=16)
                            axs[0].tick_params(labelsize=16)

                            axs[0].plot(range(0,len(logs)), loss, marker="o", c=colors[i],  linestyle='--', label="Run="+str(i+1))
                            axs[1].plot(range(0,len(logs)), acc, marker="o", c=colors[i],  linestyle='--', label="Run="+str(i+1)+" max_acc= "+str(max_acc))
        #                     axs[2].plot(range(0,len(df_results0)), acc2, marker="o", c=colors[lp],  linestyle='--', label=" Loss parameters: margin="+loss_parameters[lp])

                            axs[0].set_title("Train loss (x1000)",fontsize=16)
                            axs[1].set_title("Test accuracy Trheshold",fontsize=16)
        #                     axs[2].set_title("Test accuracy Nearest neighbor")

                            leg = axs[0].legend(loc='best', ncol=1, shadow=True, fancybox=True,fontsize=16)
                            leg = axs[1].legend(loc='best', ncol=1, shadow=True, fancybox=True,fontsize=16)
        #                     leg = axs[2].legend(loc='best', ncol=1, shadow=True, fancybox=True)
                        max_acc_avg = np.average(np.array(max_acc_list))
                        fig1.suptitle(file_name+"_Cross_validation. MAX ACC AVG = "+str(max_acc_avg),fontsize=18)
                        fig1.savefig(path+file_name+"/tmp_cv_results.png",pad_inches = 0)

# plot_cv()