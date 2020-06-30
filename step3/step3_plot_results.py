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
# loss_parameters = ["0.9+mean","0.7+mean","0.5+mean","0.4+mean","0.3+mean","0.35+mean","0.1+mean"]
loss_parameters = ["0.7+mean","0.3+mean"]
# learning_rates = [1e-1,1e-2,1.2e-2,1.5e-2,1.6e-2,2e-2,2.4e-2,4e-2,5e-2,6e-2,8e-2,1e-3,2e-3,4e-3,5e-3,6e-3,8e-3,9e-3,5e-4,7e-4]
learning_rates = [6e-2]
# splits = ["32","64","128","256","512","1024","2048","4096","4810","8192","16384","4810","9620","19240"]
splits = ["1024","2048","4096","4810","8192","16384","4810","9620","19240"]
ep_run_init = ["00"]
ep_run_end = ["150","300"]
# ep_run_init = ["00","30","50","60","100","125","150","175","200","250","300"]
# ep_run_end = ["02","50","60","70","80","100","125","150","175","200","250","300","301","302","303","400"]

#options
db_name = ["openml_203ds_datasets_matching"]
strategy = ["isolation","random"]
archi = ["Fasttext_150","Fasttext_300","Bert_300","Bert_768"]
optimizer = ["adam","sgd"]
loss_functions = ["ContrastiveLoss","CosineEmbeddingLoss"]


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
                        enot = str("{:e}".format(lr))
                        enot = enot.split("e")[0].rstrip('0').rstrip(".")+"e"+enot.split("e")[1]
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{}|epochs_run:{}_{}".format(loss_parameters[lp],s,enot,eri,ere))
                       
                        
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
                        enot = str("{:e}".format(lr))
                        enot = enot.split("e")[0].rstrip('0').rstrip(".")+"e"+enot.split("e")[1]
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{}|epochs_run:{}_{}".format(loss_parameters[lp],s,enot,eri,ere))
                        #{'epoch': 0, 'loss': 0.07563, 'acc': 0.54412, 'acc2': 0.36527, 'time_epoch': 268.2493, 'time_total': 268.2493}    
                        try:
                            df_results0 = reading(path+file_name+".txt")
                        except:
                            continue
                        loss = (np.array([x["loss"] for x in df_results0])*1000).tolist()
#                         loss = list(x["loss"] for x in df_results0)
                        acc = list(x["acc"] for x in df_results0)
                        acc2 = list(x["acc2"] for x in df_results0)

                        fig1.suptitle(str("NegSample: {}x Optimizer: {} Architecture: {} Loss: {} lr: {} splits: {}").format(sampling,op,a,lf,enot,s),fontsize=18)
                        
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


def plot_details(sampling,db,st,a,op,lf,subpath=""):
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
                        enot = str("{:e}".format(lr))
                        enot = enot.split("e")[0].rstrip('0').rstrip(".")+"e"+enot.split("e")[1]
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{}|epochs_run:{}_{}".format(loss_parameters[lp],s,enot,eri,ere))
                       
                        
                        ##sub_folders
                        if os.path.exists(path+file_name+"/"):
                            plot_details(sampling,db,st,a,op,lf,file_name+"/")
                        
                        try:
                            df_results0 = reading(path+file_name+".txt")
                            fig1, axs = plt.subplots(1, 2, figsize=(35, 10), facecolor='w', edgecolor='k')
                            enot = str("{:e}".format(lr))
                            enot = enot.split("e")[0].rstrip('0').rstrip(".")+"e"+enot.split("e")[1]
                            file_name = str("loss_parameters:{}|batch_splits:{}|lr:{}|epochs_run:{}_{}".format(loss_parameters[lp],s,enot,eri,ere))
                            #{'epoch': 0, 'loss': 0.07563, 'acc': 0.54412, 'acc2': 0.36527, 'time_epoch': 268.2493, 'time_total': 268.2493}    
                            try:
                                df_results0 = reading(path+file_name+".txt")
                            except:
                                continue
                            loss = (np.array([x["loss"] for x in df_results0])*1000).tolist()
                            acc = list(x["acc"] for x in df_results0)
                            acc2 = list(x["acc2"] for x in df_results0)
                            recall = list(x["recall"] for x in df_results0)
                            precision = list(x["precision"] for x in df_results0)
                            fscore = list(x["fscore"] for x in df_results0)

                            max_fscore,max_recall,max_prec = get_maxs(fscore,recall,precision)

                            fig1.suptitle(str("NegSample: {}x Optimizer: {} Architecture: {} Loss: {} lr: {} splits: {} Loss margin: {}").format(sampling,op,a,lf,enot,s,loss_parameters[lp]),fontsize=18)

                            axs[0].yaxis.grid(True)
                            axs[0].tick_params(labelsize=16)
                            axs[0].plot(range(0,len(df_results0)), loss, marker="o", c=colors[lp],  linestyle='--', label=" Loss parameters: margin="+loss_parameters[lp])
                            axs[0].set_title("Train loss (x1000)",fontsize=16)
                            leg = axs[0].legend(loc='best', ncol=1, shadow=True, fancybox=True,fontsize=16)

                            axs[1].set_ylim([0.5,1.0])
                            axs[1].set_yticks(np.arange(0.5, 1.05, 0.025))
                            axs[1].yaxis.grid(True)
                            axs[1].tick_params(labelsize=16)
                            axs[1].set_title("Test fscore/recall/precision",fontsize=16)

                            axs[1].plot(range(0,len(df_results0)), fscore, marker="o", c=colors[0],  linestyle='--', label="Max fscore: "+ str("{:.5f}".format(max_fscore)))
                            leg = axs[1].legend(loc='best', ncol=1, shadow=True, fancybox=True,fontsize=16)

                            axs[1].plot(range(0,len(df_results0)), recall, marker="o", c=colors[1],  linestyle='--', label="with recall: "+ str("{:.5f}".format(max_recall)))
                            leg = axs[1].legend(loc='best', ncol=1, shadow=True, fancybox=True,fontsize=16)

                            axs[1].plot(range(0,len(df_results0)), precision, marker="o", c=colors[2],  linestyle='--', label="with prec: "+ str("{:.5f}".format(max_prec)))
                            leg = axs[1].legend(loc='best', ncol=1, shadow=True, fancybox=True,fontsize=16)
                            
                            fig1.savefig(path+str("details_batch_splits:{}|lr:{:.0e}|epochs_run:{}_{}".format(s,lr,eri,ere))+".png",pad_inches = 0)  
                        except:
                            continue
        
        
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
                        enot = str("{:e}".format(lr))
                        enot = enot.split("e")[0].rstrip('0').rstrip(".")+"e"+enot.split("e")[1]
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{}|epochs_run:{}_{}".format(loss_parameters[lp],s,enot,eri,ere))
                        
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
                        enot = str("{:e}".format(lr))
                        enot = enot.split("e")[0].rstrip('0').rstrip(".")+"e"+enot.split("e")[1]
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{}|epochs_run:{}_{}".format(loss_parameters[lp],s,enot,eri,ere))
                        #{'epoch': 0, 'loss': 0.07563, 'acc': 0.54412, 'acc2': 0.36527, 'time_epoch': 268.2493, 'time_total': 268.2493}    
                        try:
                            df_results0 = reading(path+file_name+".txt")
                        except:
                            continue
                        loss = (np.array([x["loss"] for x in df_results0])*1000).tolist()
#                         loss = list(x["loss"] for x in df_results0)
                        acc = list(x["acc"] for x in df_results0)
                        acc2 = list(x["acc2"] for x in df_results0)

                        fig1.suptitle(str("NegSample: {}x Optimizer: {} Architecture: {} Loss: {} lr: {} splits: {}").format(sampling,op,a,lf,enot,s),fontsize=18)
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

def get_maxs(fscores,recalls,specs):
    max_fscore = 0.0
    max_recall = 0.0
    max_spec = 0.0
    for i in range (len(fscores)):
        if fscores[i] > max_fscore:
            max_fscore = fscores[i]
            max_recall = recalls[i]
            max_spec = specs[i]
    return max_fscore,max_recall,max_spec
        
def get_maxs_ind(fscores,recalls,specs):
    max_fscore = 0.0
    max_recall = 0.0
    max_spec = 0.0
    for i in range (len(fscores)):
        if fscores[i] > max_fscore:
            max_fscore = fscores[i]
        if recalls[i] > max_recall:
            max_recall = recalls[i]
        if specs[i] > max_spec:
            max_spec = specs[i]
    return max_fscore,max_recall,max_spec    
        
def get_max_acc(accs):
    max_acc = 0.0
    for acc in accs:
        if acc > max_acc:
            max_acc = acc
    return max_acc
    
def plot_cv_3(sampling,db,st,a,op,lf,subpath=""):
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
                        enot = str("{:e}".format(lr))
                        enot = enot.split("e")[0].rstrip('0').rstrip(".")+"e"+enot.split("e")[1]
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{}|epochs_run:{}_{}".format(loss_parameters[lp],s,enot,eri,ere))

                        ##sub_folders
                        if os.path.exists(path+file_name+"/"):
                            plot_cv_3(sampling,db,st,a,op,lf,file_name+"/")
                                
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
                        enot = str("{:e}".format(lr))
                        enot = enot.split("e")[0].rstrip('0').rstrip(".")+"e"+enot.split("e")[1]
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{}|epochs_run:{}_{}".format(loss_parameters[lp],s,enot,eri,ere))
                        
                        df_results0 = []
                        for cv_i in range(10):                            
                            try:
                                df_ = reading(path+file_name+"/tmp_cv_result_"+str(cv_i)+".txt")
                            except:
                                continue
                            df_results0.append((cv_i,df_))
                        max_acc_list = []
                        for i in  range (len(df_results0)):
                            logs = df_results0[i][1]
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

                            axs[0].plot(range(0,len(logs)), loss, marker="o", c=colors[df_results0[i][0]],  linestyle='--', label="Run="+str(df_results0[i][0]+1))
                            axs[1].plot(range(0,len(logs)), acc, marker="o", c=colors[df_results0[i][0]],  linestyle='--', label="Run="+str(df_results0[i][0]+1)+" max_acc= "+str("{:.5f}".format(max_acc)))
        #                     axs[2].plot(range(0,len(df_results0)), acc2, marker="o", c=colors[lp],  linestyle='--', label=" Loss parameters: margin="+loss_parameters[lp])

                            axs[0].set_title("Train loss (x1000)",fontsize=16)
                            axs[1].set_title("Test accuracy Trheshold",fontsize=16)
        #                     axs[2].set_title("Test accuracy Nearest neighbor")

                            leg = axs[0].legend(loc='best', ncol=1, shadow=True, fancybox=True,fontsize=16)
                            leg = axs[1].legend(loc='best', ncol=1, shadow=True, fancybox=True,fontsize=16)
        #                     leg = axs[2].legend(loc='best', ncol=1, shadow=True, fancybox=True)
                        max_acc_avg = np.average(np.array(max_acc_list))
                        fig1.suptitle(a+"-"+file_name+"_Cross_validation. MAX ACC AVG = "+str(max_acc_avg),fontsize=18)
                        fig1.savefig(path+file_name+"/tmp_cv_results.png",pad_inches = 0)

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
                    for lp in range(len(loss_parameters)): 
                        enot = str("{:e}".format(lr))
                        enot = enot.split("e")[0].rstrip('0').rstrip(".")+"e"+enot.split("e")[1]
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{}|epochs_run:{}_{}".format(loss_parameters[lp],s,enot,eri,ere))

                        ##sub_folders
                        if os.path.exists(path+file_name+"/"):
                            plot_cv_3(sampling,db,st,a,op,lf,file_name+"/")
                                
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

                        fig1, axs = plt.subplots(1, 3, figsize=(35, 10), facecolor='w', edgecolor='k')
                        enot = str("{:e}".format(lr))
                        enot = enot.split("e")[0].rstrip('0').rstrip(".")+"e"+enot.split("e")[1]
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{}|epochs_run:{}_{}".format(loss_parameters[lp],s,enot,eri,ere))
                        
                        df_results0 = []
                        for cv_i in range(10):                            
                            try:
                                df_ = reading(path+file_name+"/tmp_cv_result_"+str(cv_i)+".txt")
                            except:
                                continue
                            df_results0.append((cv_i,df_))
                        max_fscore_list = []
                        max_recall_list = []
                        max_prec_list = []
                        for i in  range (len(df_results0)):
                            logs = df_results0[i][1]
                            run = df_results0[i][0]
                            loss = (np.array([x["loss"] for x in logs])*1000).tolist()
                            acc = list(x["acc"] for x in logs)
                            acc2 = list(x["acc2"] for x in logs)
                            recall = list(x["recall"] for x in logs)
                            precision = list(x["precision"] for x in logs)
                            fscore = list(x["fscore"] for x in logs)
                            
                            max_fscore = get_maxs(fscore,recall,precision)
                            max_recall = get_maxs(recall,precision,fscore)
                            max_prec = get_maxs(precision,recall,fscore)
                            max_fscore_list.append(max_fscore[0])
                            max_recall_list.append(max_recall[0])
                            max_prec_list.append(max_prec[0])
                            
                            axs[0].set_ylim([0.5,1.0])
                            axs[0].set_yticks(np.arange(0.5, 1.05, 0.05))
                            axs[0].yaxis.grid(True)
                            axs[0].tick_params(labelsize=16)
                            axs[0].set_title("Test Fscore",fontsize=16)
                            axs[0].plot(range(0,len(fscore)), fscore, marker="o", c=colors[run],  linestyle='--', label=str("Run:{:.0f}, Max_Fsc: {:.5f}, with Rec: {:.5f}, Prec: {:.5f}".format(df_results0[i][0]+1,max_fscore[0],max_fscore[1],max_fscore[2])))
                            
                            axs[1].set_ylim([0.5,1.0])
                            axs[1].set_yticks(np.arange(0.5, 1.05, 0.05))
                            axs[1].yaxis.grid(True)
                            axs[1].tick_params(labelsize=16)
                            axs[1].set_title("Test Recall",fontsize=16)
                            axs[1].plot(range(0,len(recall)), recall, marker="o", c=colors[run],  linestyle='--', label=str("Run:{:.0f}, Max_Rec: {:.5f}, with Prec: {:.5f}, Fsc: {:.5f}".format(df_results0[i][0]+1,max_recall[0],max_recall[1],max_recall[2])))
                            
                            axs[2].set_ylim([0.5,1.0])
                            axs[2].set_yticks(np.arange(0.5, 1.05, 0.05))
                            axs[2].yaxis.grid(True)
                            axs[2].tick_params(labelsize=16)
                            axs[2].plot(range(0,len(precision)), precision, marker="o", c=colors[run],  linestyle='--', label=str("Run:{:.0f}, Max_Prec: {:.5f}, with Rec: {:.5f}, Fsc: {:.5f}".format(df_results0[i][0]+1,max_prec[0],max_prec[1],max_prec[2])))
        
                        max_fscore_avg = np.average(np.array(max_fscore_list))
                        max_recall_avg = np.average(np.array(max_recall_list))
                        max_prec_avg = np.average(np.array(max_prec_list))
                        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1, shadow=True, fancybox=True,fontsize=16)
                        axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1, shadow=True, fancybox=True,fontsize=16)
                        axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1, shadow=True, fancybox=True,fontsize=16)
                        axs[0].set_title(str("Test Fscore AVG: {:.5f}".format(max_fscore_avg)),fontsize=16)
                        axs[1].set_title(str("Test Recall AVG: {:.5f}".format(max_recall_avg)),fontsize=16)
                        axs[2].set_title(str("Test Precision AVG: {:.5f}".format(max_prec_avg)),fontsize=16)
                        fig1.suptitle("{}-{}_Cross_validation".format(a,file_name))
                        fig1.savefig(path+file_name+"/tmp_cv_results.png",pad_inches = 0)

                        
def plot_cv_details(sampling,db,st,a,op,lf,subpath=""):
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
                        enot = str("{:e}".format(lr))
                        enot = enot.split("e")[0].rstrip('0').rstrip(".")+"e"+enot.split("e")[1]
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{}|epochs_run:{}_{}".format(loss_parameters[lp],s,enot,eri,ere))

                        ##sub_folders
                        if os.path.exists(path+file_name+"/"):
                            plot_cv_3(sampling,db,st,a,op,lf,file_name+"/")
                                
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

                        try:
                            fig1, axs = plt.subplots(1, 2, figsize=(35, 10), facecolor='w', edgecolor='k')
                            enot = str("{:e}".format(lr))
                            enot = enot.split("e")[0].rstrip('0').rstrip(".")+"e"+enot.split("e")[1]
                            file_name = str("loss_parameters:{}|batch_splits:{}|lr:{}|epochs_run:{}_{}".format(loss_parameters[lp],s,enot,eri,ere))

                            df_results0 = []
                            for cv_i in range(10):                            
                                try:
                                    df_ = reading(path+file_name+"/tmp_cv_result_"+str(cv_i)+".txt")
                                except:
                                    continue
                                df_results0.append((cv_i,df_))
                            max_fscore_list = []
                            max_recall_list = []
                            max_prec_list = []
                            for i in  range (len(df_results0)):
                                logs = df_results0[i][1]
                                run = df_results0[i][0]
                                loss = (np.array([x["loss"] for x in logs])*1000).tolist()
                                acc = list(x["acc"] for x in logs)
                                acc2 = list(x["acc2"] for x in logs)
                                recall = list(x["recall"] for x in logs)
                                precision = list(x["precision"] for x in logs)
                                fscore = list(x["fscore"] for x in logs)

                                max_fscore = get_maxs(fscore,recall,precision)
                                max_recall = get_maxs(recall,precision,fscore)
                                max_prec = get_maxs(precision,recall,fscore)
                                max_fscore_list.append(max_fscore[0])
                                max_recall_list.append(max_recall[0])
                                max_prec_list.append(max_prec[0])

    #                             axs[0].set_ylim([0.5,1.0])
    #                             axs[0].set_yticks(np.arange(0.5, 1.05, 0.05))
    #                             axs[0].yaxis.grid(True)
                                axs[0].tick_params(labelsize=16)
                                axs[0].plot(range(0,len(loss)), loss, marker="o", c=colors[run],  linestyle='--', label=str("Run:{:.0f}".format(df_results0[i][0]+1)))

#                                 axs[1].set_ylim([0.5,1.0])
#                                 axs[1].set_yticks(np.arange(0.5, 1.05, 0.05))
                                axs[1].set_ylim([0.0,1.0])
                                axs[1].set_yticks(np.arange(0.0, 1.05, 0.05))
                                axs[1].yaxis.grid(True)
                                axs[1].tick_params(labelsize=16)
                                axs[1].plot(range(0,len(fscore)), fscore, marker="o", c=colors[run],  linestyle='--', label=str("Run:{:.0f}, Max Fscore: {:.5f}, with Recall: {:.5f} and Precision: {:.5f}".format(df_results0[i][0]+1,max_fscore[0],max_fscore[1],max_fscore[2])))
                        except:
                            continue
                        
                        max_fscore_avg = np.average(np.array(max_fscore_list))
                        var_max_fscore = np.var(np.array(max_fscore_list))
                        std_max_fscore = np.std(np.array(max_fscore_list))
                        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1, shadow=True, fancybox=True,fontsize=16)
                        axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1, shadow=True, fancybox=True,fontsize=16)
                        axs[0].set_title("Train Loss (x1000)",fontsize=16)
                        axs[1].set_title(str("Test Fscore AVG: {:.5f} - Var: {:.5f} - SD: {:.5f}".format(max_fscore_avg,var_max_fscore,std_max_fscore)),fontsize=16)
                        fig1.suptitle("{}-{}_Cross_validation".format(a,lf+"_"+file_name),fontsize=18)
                        fig1.savefig(path+file_name+"/tmp_cv_results.png",pad_inches = 0)
                        