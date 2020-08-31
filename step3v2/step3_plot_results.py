import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

def reading(file_path):
    s = open(file_path, 'r').read()
    return eval(s)

#parameters for iterations
colors = []
for i in range (5):
    colors+=["red","green","blue","orange","black","pink","grey","skyblue","yellow","brown","#d4b0b3","#d4b0d3","#bdb0d4","#00fff8","#00ff1a", "#e1ff00", "#960b0b","#76960b","#cef76d","#00044d"]
loss_parameters = ["0.5+mean"]
learning_rates = [1e-2,6e-3]
splits = ["512","1024","4096","2048","40000"]
ep_run_init = ["00"]
ep_run_end = ["50"]

#options
db_name = ["openml_203ds_datasets_matching"]
strategy = ["isolation","random"]
archi = ["Fasttext_150","Fasttext_300","Bert_300","Bert_768"]
optimizer = ["adam","sgd"]
loss_functions = ["ContrastiveLoss","CosineEmbeddingLoss"]


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


def plot_details(sampling,db,st,a,op,lf,subpath="",nit=None):

    if nit==None:
        epochs=ep_run_end
    else:
        epochs=nit
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
                            max_acc = get_max_acc(acc)

#                             fig1.suptitle(str("NegSample: {}x Optimizer: {} Architecture: {} Loss: {} lr: {} splits: {} Loss margin: {}").format(sampling,op,a,lf,enot,s,loss_parameters[lp]),fontsize=18)

                            axs[0].set_ylim([0.0,1.0])
                            axs[0].set_yticks(np.arange(0.0, 1.05, 0.05))
                            axs[0].yaxis.grid(True)
                            axs[0].tick_params(labelsize=18)
                            axs[0].plot(range(0,len(df_results0)), acc, marker="o", c=colors[lp],  linestyle='--', label="Max accuracy: "+ str("{:.3f}".format(max_acc)))
                            axs[0].set_title("Test accuracy",fontsize=24)
                            leg = axs[0].legend(loc='best', ncol=1, shadow=True, fancybox=True,fontsize=24)

                            axs[1].set_ylim([0.0,1.0])
                            axs[1].set_yticks(np.arange(0.0, 1.05, 0.05))
                            axs[1].yaxis.grid(True)
                            axs[1].tick_params(labelsize=18)
                            axs[1].set_title("Test fscore/recall/precision",fontsize=24)

                            axs[1].plot(range(0,len(df_results0)), fscore, marker="o", c=colors[0],  linestyle='--', label="Max fscore: "+ str("{:.3f}".format(max_fscore)))
                            leg = axs[1].legend(loc='best', ncol=1, shadow=True, fancybox=True,fontsize=24)

                            axs[1].plot(range(0,len(df_results0)), recall, marker="o", c=colors[1],  linestyle='--', label="with recall: "+ str("{:.3f}".format(max_recall)))
                            leg = axs[1].legend(loc='best', ncol=1, shadow=True, fancybox=True,fontsize=24)

                            axs[1].plot(range(0,len(df_results0)), precision, marker="o", c=colors[2],  linestyle='--', label="with prec: "+ str("{:.3f}".format(max_prec)))
                            leg = axs[1].legend(loc='best', ncol=1, shadow=True, fancybox=True,fontsize=24)
                            
                            fig1.savefig(path+str("details_batch_splits:{}|lr:{:.0e}|epochs_run:{}_{}".format(s,lr,eri,ere))+".png",pad_inches = 0)  
                        except:
                            continue
        
            
                        
def plot_cv_details(sampling,db,st,a,op,lf,subpath="",nit=None):
    ##options selection
#     sampling = 2
#     db = db_name[0]
#     st = strategy[1]
#     a = archi[0]
#     op = optimizer[0]
#     lf = loss_functions[0]
    if nit==None:
        epochs=ep_run_end
    else:
        epochs=nit
    path = str("./results/{}/{}/{}/cv/net_name:{}/optimizer_name:{}/loss_name:{}/".format(db,st,sampling,a,op,lf))
    path += subpath
    for s in splits:
        for lr in learning_rates:
            for eri in ep_run_init:
                for ere in epochs:
                    for lp in range(len(loss_parameters)): 
                        enot = str("{:e}".format(lr))
                        enot = enot.split("e")[0].rstrip('0').rstrip(".")+"e"+enot.split("e")[1]
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{}|epochs_run:{}_{}".format(loss_parameters[lp],s,enot,eri,ere))

                        ##sub_folders
                        if os.path.exists(path+file_name+"/"):
                            plot_cv_details(sampling,db,st,a,op,lf,file_name+"/")
                                
                        content = False
                        for cv_i in range(89):
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
                            for cv_i in range(89):                            
                                try:
                                    df_ = reading(path+file_name+"/tmp_cv_result_"+str(cv_i)+".txt")
                                except:
                                    continue
                                df_results0.append((cv_i,df_))
                            max_fscore_list = []
                            max_recall_list = []
                            max_prec_list = []
                            max_acc_list = []
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
                                max_acc = get_max_acc(acc)
                                max_fscore_list.append(max_fscore[0])
                                max_recall_list.append(max_fscore[1])
                                max_prec_list.append(max_fscore[2])
                                max_acc_list.append(max_acc)

                                axs[0].set_ylim([0.0,1.0])
                                axs[0].set_yticks(np.arange(0.0, 1.05, 0.05))
                                axs[0].yaxis.grid(True)
                                axs[0].tick_params(labelsize=18)
                                axs[0].plot(range(0,len(loss)), acc, marker="o", c=colors[i],  linestyle='--', label="Run:{:.0f},Max Accuracy: {:.5f}".format(df_results0[i][0]+1,max_acc))
                                #format(df_results0[i][0]

                                axs[1].set_ylim([0.0,1.0])
                                axs[1].set_yticks(np.arange(0.0, 1.05, 0.05))
                                axs[1].yaxis.grid(True)
                                axs[1].tick_params(labelsize=18)
                                axs[1].plot(range(0,len(fscore)), fscore, marker="o", c=colors[i],  linestyle='--', label=str("Run:{:.0f}, Max Fscore: {:.5f}, with Recall: {:.5f} and Precision: {:.5f}".format(df_results0[i][0]+1,max_fscore[0],max_fscore[1],max_fscore[2])))
                        #df_results0[i][0]        
                        except:
                            continue
                        
                        max_fscore_avg = np.average(np.array(max_fscore_list))
                        var_max_fscore = np.var(np.array(max_fscore_list))
                        std_max_fscore = np.std(np.array(max_fscore_list))
                        
                        max_recall_avg = np.average(np.array(max_recall_list))
                        var_max_recall = np.var(np.array(max_recall_list))
                        std_max_recall = np.std(np.array(max_recall_list))
                        
                        max_prec_avg = np.average(np.array(max_prec_list))
                        var_max_prec = np.var(np.array(max_prec_list))
                        std_max_prec = np.std(np.array(max_prec_list))
                        
                        max_acc_avg = np.average(np.array(max_acc_list))
                        var_max_acc = np.var(np.array(max_acc_list))
                        std_max_acc = np.std(np.array(max_acc_list))
                        
                        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1, shadow=True, fancybox=True,fontsize=24)
#                         axs[0].set_title(str("Test accuracy AVG: {:.5f} - Var: {:.5f} - SD: {:.5f}".format(max_acc_avg,var_max_acc,std_max_acc)),fontsize=18)
                        axs[0].set_title(str("Test accuracy AVG: {:.3f} - SD: {:.3f}".format(max_acc_avg,std_max_acc)),fontsize=24)
                        axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1, shadow=True, fancybox=True,fontsize=24)
#                         axs[0].set_title("Train Loss (x1000)",fontsize=16)
#                         axs[1].set_title(str("Test Fscore AVG: {:.5f} - Var: {:.5f} - SD: {:.5f} ; Test Recall AVG: {:.5f} - Var: {:.5f} - SD: {:.5f} ; Test Precision AVG: {:.5f} - Var: {:.5f} - SD: {:.5f}".format(max_fscore_avg,var_max_fscore,std_max_fscore,max_recall_avg,var_max_recall,std_max_recall,max_prec_avg,var_max_prec,std_max_prec)),fontsize=18)
                        axs[1].set_title(str("AVG Test: Fscore: {:.3f} - SD: {:.3f} ; Recall: {:.3f} - SD: {:.3f} ; Precision: {:.3f} - SD: {:.3f}".format(max_fscore_avg,std_max_fscore,max_recall_avg,std_max_recall,max_prec_avg,std_max_prec)),fontsize=24)
#                         fig1.suptitle("{}-{}_Cross_validation".format(a,lf+"_"+file_name),fontsize=18)
                        fig1.suptitle("")
                        fig1.savefig(path+file_name+"/tmp_cv_results.png",pad_inches = 0)
                        
                  
def plot_bar(sampling,db,st,a,op,lf,subpath="",nit=None):
    if nit==None:
        epochs=ep_run_end
    else:
        epochs=nit
    path = str("./results/{}/{}/{}/cv/net_name:{}/optimizer_name:{}/loss_name:{}/".format(db,st,sampling,a,op,lf))
    path += subpath
    for s in splits:
        for lr in learning_rates:
            for eri in ep_run_init:
                for ere in epochs:
                    for lp in range(len(loss_parameters)): 
                        enot = str("{:e}".format(lr))
                        enot = enot.split("e")[0].rstrip('0').rstrip(".")+"e"+enot.split("e")[1]
                        file_name = str("loss_parameters:{}|batch_splits:{}|lr:{}|epochs_run:{}_{}".format(loss_parameters[lp],s,enot,eri,ere))

                        ##sub_folders
                        if os.path.exists(path+file_name+"/"):
                            plot_cv_details(sampling,db,st,a,op,lf,file_name+"/")
                                
                        content = False
                        for cv_i in range(89):
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
                            enot = str("{:e}".format(lr))
                            enot = enot.split("e")[0].rstrip('0').rstrip(".")+"e"+enot.split("e")[1]
                            file_name = str("loss_parameters:{}|batch_splits:{}|lr:{}|epochs_run:{}_{}".format(loss_parameters[lp],s,enot,eri,ere))

                            df_results0 = []
                            for cv_i in range(89):                            
                                try:
                                    df_ = reading(path+file_name+"/tmp_cv_result_"+str(cv_i)+".txt")
                                except:
                                    continue
                                df_results0.append((cv_i,df_))
                            max_fscore_list = []
                            max_recall_list = []
                            max_prec_list = []
                            max_acc_list = []
                            
                            width = 0.35  # the width of the bars
                            fig1, ax = plt.subplots(figsize=(20, 5))
#                             fig1, ax = plt.subplots(width=len(df_results0))
                            
                            labels = []
                            
                            for i in  range (len(df_results0)):
                                labels.append(str(i+1))
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
                                max_acc = get_max_acc(acc)
                                max_fscore_list.append(max_fscore[0])
                                max_recall_list.append(max_fscore[1])
                                max_prec_list.append(max_fscore[2])
                                max_acc_list.append(max_acc)
                                
                        except:
                            continue
                        
                        
                        x = np.arange(len(labels))  # the label locations
                        
#                         rects1 = ax.bar(x - 3*width/4, max_fscore_list, width/2, label='Fscore')
#                         rects2 = ax.bar(x - width/4, max_recall_list, width/2, label='Recall')
#                         rects3 = ax.bar(x + width/4, max_prec_list, width/2, label='Precision')
#                         rects0 = ax.bar(x + 3*width/4, max_acc_list, width/2, label='Accuracy')

                        rects1 = ax.bar(x - 3*width/4, max_acc_list, width/2, label='Max Accuracy')
                        rects2 = ax.bar(x - width/4, max_fscore_list, width/2, label='Max Fscore')
                        rects3 = ax.bar(x + width/4, max_recall_list, width/2, label='Max Recall')
                        rects4 = ax.bar(x + 3*width/4, max_acc_list, width/2, label='Max Precision')

                        max_fscore_avg = np.average(np.array(max_fscore_list))
                        var_max_fscore = np.var(np.array(max_fscore_list))
                        std_max_fscore = np.std(np.array(max_fscore_list))
                        
                        max_recall_avg = np.average(np.array(max_recall_list))
                        var_max_recall = np.var(np.array(max_recall_list))
                        std_max_recall = np.std(np.array(max_recall_list))
                        
                        max_prec_avg = np.average(np.array(max_prec_list))
                        var_max_prec = np.var(np.array(max_prec_list))
                        std_max_prec = np.std(np.array(max_prec_list))
                        
                        max_acc_avg = np.average(np.array(max_acc_list))
                        var_max_acc = np.var(np.array(max_acc_list))
                        std_max_acc = np.std(np.array(max_acc_list))
                        
                        # Add some text for labels, title and custom x-axis tick labels, etc.
#                         axs[0].set_ylim([0.0,1.0])
                        ax.set_yticks(np.arange(0.0, 1.05, 0.05))
                        ax.yaxis.grid(True)
                        ax.set_ylabel('Score')
                        ax.set_title("Test accuracy AVG: {:.3f} - SD: {:.3f} ; AVG Test: Fscore: {:.3f} - SD: {:.3f} ; Recall: {:.3f} - SD: {:.3f} ; Precision: {:.3f} - SD: {:.3f}".format(max_acc_avg,std_max_acc,max_fscore_avg,std_max_fscore,max_recall_avg,std_max_recall,max_prec_avg,std_max_prec))
                        ax.set_xticks(x)
                        ax.set_xticklabels(labels)
                        ax.legend()

                        fig1.savefig(path+file_name+"/bar_cv_results.png",bbox_inches="tight",dpi=300)