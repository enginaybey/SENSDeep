# -*- coding: utf-8 -*-
"""
@author: engin aybey
"""

import pandas as pd
# Setup plotting
import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
# # Set Matplotlib defaults
# plt.rc('figure', autolayout=True)
# plt.rc('axes', labelweight='bold', labelsize='large',
#        titleweight='bold', titlesize=18, titlepad=10)
# plt.rc('animation', html='html5')
import numpy as np
import metrics_eval as mtr
import os,sys
import time,math,random,gc,downdataset,joblib
from reportwrite import writeclassreport, writeclassreporta
from npztonp import npztonp
# from BahdanauAttention import AttentionLayer
from contextlib import redirect_stdout
from sklearn.utils import class_weight
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import models, regularizers
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout
from tensorflow.keras.layers import concatenate, add
from argparse import ArgumentParser,ArgumentTypeError
from dlayers.attention import AttentionLayer
from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric
import lightgbm as lgb
import xgboost as xgb
from math import log
def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise ArgumentTypeError(f"readable_dir:{path} is not a valid path")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
parser = ArgumentParser()
print(sys.argv[0])
parser.add_argument('-p','--path', type=dir_path)
parser.add_argument('-p2','--path2', type=dir_path,required=False,default=False)
parser.add_argument('-us','--undersamp', type=str_to_bool, nargs='?', const=True, default=False)    
parser.add_argument('-ad','--alldata', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('-nvd','--novaldtn', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('-pdb','--pdbsets', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('-cw','--classw', type=str_to_bool, nargs='?', const=True, default=False)   
parser.add_argument('-cwc','--classwc', type=str_to_bool, nargs='?', const=True, default=False) 
parser.add_argument('-sq','--sqrtw', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('-sw','--sampw', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('-sc','--scale_data', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument("-st", "--scaler_type", type=str, required=False, default=None)
parser.add_argument("-dv", "--device_type", type=str, required=False, default="CPU")
parser.add_argument("-fn", "--fld_no", type=str, required=False, default="-1")
parser.add_argument("-md", "--model_no", type=str, required=False, default="-1")
parser.add_argument("-ws", "--wsize", type=str, nargs='?', const="", default="")

args = parser.parse_args()
undersampling=args.undersamp
all_data=args.alldata
pdbsets=args.pdbsets
classw=args.classw
classwcomp=args.classwc
sqrtw=args.sqrtw
sampw=args.sampw
npzfiles_dir=args.path
testfiles_dir=args.path2
device_type=args.device_type
novald=args.novaldtn
scale_data=args.scale_data
scaler_type=args.scaler_type
fld_no=args.fld_no
model_no=args.model_no
wsize=args.wsize

def load_data(under_lbl):
    # npzfiles_dir=sys.argv[1]
    X_train=npztonp(np.load(npzfiles_dir+'X_train'+under_lbl+'.npz'))
    y_train=npztonp(np.load(npzfiles_dir+'y_train'+under_lbl+'.npz'))
    if under_lbl!="_all": 
        X_valid=npztonp(np.load(npzfiles_dir+'X_valid'+under_lbl+'.npz'))
        y_valid=npztonp(np.load(npzfiles_dir+'y_valid'+under_lbl+'.npz'))
    else:
        X_valid=np.array([])
        y_valid=np.array([])
    if pdbsets==True:
        X_72=npztonp(np.load(npzfiles_dir+'X_72.npz'))
        y_72=npztonp(np.load(npzfiles_dir+'y_72.npz'))
        X_164=npztonp(np.load(npzfiles_dir+'X_164.npz'))
        y_164=npztonp(np.load(npzfiles_dir+'y_164.npz'))
        X_186=npztonp(np.load(npzfiles_dir+'X_186.npz'))
        y_186=npztonp(np.load(npzfiles_dir+'y_186.npz'))
    else:
        X_72=np.array([])
        y_72=np.array([])
        X_164=np.array([])
        y_164=np.array([])
        X_186=np.array([])
        y_186=np.array([])
    X_355=npztonp(np.load(npzfiles_dir+'X_355.npz'))
    y_355=npztonp(np.load(npzfiles_dir+'y_355.npz'))
    X_448=npztonp(np.load(npzfiles_dir+'X_448.npz'))
    y_448=npztonp(np.load(npzfiles_dir+'y_448.npz'))
    return [X_train, y_train, X_valid, y_valid, \
            X_72, y_72,  X_164, y_164,  X_186, y_186, \
                X_355, y_355, X_448, y_448]
def load_data2(under_lbl):
    # npzfiles_dir=sys.argv[1]
    indx=[0,1,5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 21, 22, 24, 26, 27, 32, 34, 35, 36, 37, 38]
    X_train=npztonp(np.load(npzfiles_dir+'X_train'+under_lbl+'.npz'))
    X_train=X_train[:,indx]
    y_train=npztonp(np.load(npzfiles_dir+'y_train'+under_lbl+'.npz'))
    X_valid=npztonp(np.load(npzfiles_dir+'X_valid'+under_lbl+'.npz'))
    X_valid=X_valid[:,indx]
    y_valid=npztonp(np.load(npzfiles_dir+'y_valid'+under_lbl+'.npz'))
    X_355=npztonp(np.load(npzfiles_dir+'X_355.npz'))
    X_355=X_355[:,indx]
    y_355=npztonp(np.load(npzfiles_dir+'y_355.npz'))
    X_448=npztonp(np.load(npzfiles_dir+'X_448.npz'))
    X_448=X_448[:,indx]
    y_448=npztonp(np.load(npzfiles_dir+'y_448.npz'))
    return X_train, y_train, X_valid, y_valid, \
            X_355, y_355, X_448, y_448

def load_dataML(under_lbl,wsize):
    df = pd.read_csv(npzfiles_dir+'/df_alldata'+wsize+'_training.csv', index_col=0, header=0)
    # df = df[0:10000]
    if under_lbl=="_down": df=downdataset.down_dataset_1_1(df)
    X=df.drop(["Interaction"],axis=1)
    y=df["Interaction"]
    if under_lbl!="_all":
        X_train, X_valid, y_train, y_valid = train_test_split(X, y,stratify=y,test_size=0.2,random_state=42)
        X_valid.columns=X.columns
        y_valid.name=y.name
        X_train.columns=X.columns
        y_train.name=y.name
    else:
        X_train=X
        y_train=y
        X_valid=pd.DataFrame([])
        y_valid=pd.DataFrame([])
    if pdbsets==True:
        df_td = pd.read_csv(testfiles_dir+"/df_alldata"+wsize+"_Dset_72.csv", index_col=0, header=0)
        X_72=df_td.iloc[:,:-1]
        y_72=df_td.iloc[:,-1]
        df_td = pd.read_csv(testfiles_dir+"/df_alldata"+wsize+"_Dset_164.csv", index_col=0, header=0)
        X_164=df_td.iloc[:,:-1]
        y_164=df_td.iloc[:,-1]
        df_td = pd.read_csv(testfiles_dir+"/df_alldata"+wsize+"_Dset_186.csv", index_col=0, header=0)
        X_186=df_td.iloc[:,:-1]
        y_186=df_td.iloc[:,-1]
    else:
        X_72=pd.DataFrame([])
        y_72=pd.DataFrame([])
        X_164=pd.DataFrame([])
        y_164=pd.DataFrame([])
        X_186=pd.DataFrame([])
        y_186=pd.DataFrame([])
    df_td = pd.read_csv(testfiles_dir+"/df_alldata"+wsize+"_Dset_355.csv", index_col=0, header=0)
    X_355=df_td.iloc[:,:-1]
    y_355=df_td.iloc[:,-1]
    df_td = pd.read_csv(testfiles_dir+"/df_alldata"+wsize+"_Dset_448.csv", index_col=0, header=0)
    X_448=df_td.iloc[:,:-1]
    y_448=df_td.iloc[:,-1]
    return [X_train, y_train, X_valid, y_valid, \
            X_72, y_72,  X_164, y_164,  X_186, y_186, \
                X_355, y_355, X_448, y_448]

def select_testsets(pdbsets,Xy_datasets):
    if pdbsets==True: 
        return {"X_72":Xy_datasets[4:6],"X_164":Xy_datasets[6:8],
                    "X_186":Xy_datasets[8:10],"X_355":Xy_datasets[10:12],
                    "X_448":Xy_datasets[12:14]}
    else:
        return {"X_355":Xy_datasets[10:12],"X_448":Xy_datasets[12:14]}
    
def plot_model_history(history,model_dir):
    history_df = pd.DataFrame(history.history)
    # print(history_df)
    history_df.to_csv(model_dir+'/model_history.csv')
    history_df.index += 1 
    if 'AUC' in history_df and 'val_AUC' in history_df: 
        history_df.rename(columns={"AUC": "auc", "val_AUC": "val_auc"},inplace=True)
    # print(history_df)
    if 'loss' in history_df and 'val_loss' in history_df:
        fig_loss=history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy loss vs Epoch").get_figure()
        if len(history_df)<=15: 
            plt.xticks(range(1,len(history_df)+1))
        else:
            plt.xticks([1]+[i for i in range(5,len(history_df)+1,5)])
        plt.xlabel('Epoch', fontweight="normal")
        plt.ylabel('Loss', fontweight="normal")
        fig_loss.savefig(model_dir+'/cross_entropy.jpg')
        plt.show()
        plt.close()
    if 'auprc' in history_df and 'val_auprc' in history_df:
        fig_auc=history_df.loc[:, ['auprc', 'val_auprc']].plot(title="AUPRC vs Epoch").get_figure()
        if len(history_df)<=15: 
            plt.xticks(range(1,len(history_df)+1))
        else:
            plt.xticks([1]+[i for i in range(5,len(history_df)+1,5)])
        plt.xlabel('Epoch', fontweight="normal")
        plt.ylabel('AUC', fontweight="normal")
        fig_auc.savefig(model_dir+'/auprc.jpg')
        plt.show()
        plt.close()
    if 'auroc' in history_df and 'val_auroc' in history_df:
        fig_auc=history_df.loc[:, ['auroc', 'val_auroc']].plot(title="AUROC vs Epoch").get_figure()
        if len(history_df)<=15: 
            plt.xticks(range(1,len(history_df)+1))
        else:
            plt.xticks([1]+[i for i in range(5,len(history_df)+1,5)])
        plt.xlabel('Epoch', fontweight="normal")
        plt.ylabel('AUC', fontweight="normal")
        fig_auc.savefig(model_dir+'/auroc.jpg')
        plt.show()
        plt.close()
    if 'binary_accuracy' in history_df and 'val_binary_accuracy' in history_df:
        fig_acc=history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy vs Epoch").get_figure()
        if len(history_df)<=15: 
            plt.xticks(range(1,len(history_df)+1))
        else:
            plt.xticks([1]+[i for i in range(5,len(history_df)+1,5)])
        plt.xlabel('Epoch', fontweight="normal")
        plt.ylabel('Accuracy', fontweight="normal")
        fig_acc.savefig(model_dir+'/accuracy.jpg')
        plt.show()
        plt.close()
        
def testset_prediction_eval3(model,X,y,X_arr,model_dir,metrics_results,testset_lbl,scale_data,scaler,clsf_model):
    if scale_data==True:
        for i in range(len(X)):
            X[i]=scaler.fit_transform(X[i])
        if clsf_model=="CNN" and len(X.shape)!=4: X = np.reshape(X, (X.shape[0],X.shape[1],X.shape[2],1))
    if len(X_arr)!=0:
        if scale_data==True:
            for i in range(len(X_arr)):
                for j in range(len(X_arr[i])):
                    X_arr[i][j]=scaler.transform(X_arr[i][j])
            if len(X[1].shape)!=4:
                X_arr[1] = np.reshape(X_arr[1], (X_arr[1].shape[0],X_arr[1].shape[1],X_arr[1].shape[2],1))
        probas,_,_ = model.predict(X_arr)
        probas=probas.ravel()
    else:
        probas,_,_ = model.predict(X)
        probas=probas.ravel()
    labels = (probas >= 0.5).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    mtr.plot_metrics_keras(y,y_pred,probas,testset_lbl,model_dir)
    print(confusion_matrix(y,y_pred))
    if len(X_arr)!=0: 
        metr=mtr.df_metrics(model,X_arr,y,y_pred,testset_lbl,y_probas=probas)
    else:
        metr=mtr.df_metrics(model,X,y,y_pred,testset_lbl,y_probas=probas)
    metrics_results.append(metr)
    print(classification_report(y,y_pred,digits=3))
    writeclassreporta(y,y_pred,"Dtestset "+testset_lbl[2:]+" \n", model_dir+'/classreport.txt')

def testset_prediction_eval4(model,X,y,X_arr,model_dir,metrics_results,testset_lbl,scale_data,scaler,clsf_model):
    if scale_data==True:
        for i in range(len(X)):
            X[i]=scaler.fit_transform(X[i])
        if clsf_model=="CNN" and len(X.shape)!=4: X = np.reshape(X, (X.shape[0],X.shape[1],X.shape[2],1))
    if len(X_arr)!=0:
        if scale_data==True:
            for i in range(len(X_arr)):
                for j in range(len(X_arr[i])):
                    X_arr[i][j]=scaler.transform(X_arr[i][j])
            if len(X[1].shape)!=4:
                X_arr[1] = np.reshape(X_arr[1], (X_arr[1].shape[0],X_arr[1].shape[1],X_arr[1].shape[2],1))
        probas,_ = model.predict(X_arr)
        probas=probas.ravel()
    else:
        probas,_= model.predict(X)
        probas=probas.ravel()
    labels = (probas >= 0.5).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    mtr.plot_metrics_keras(y,y_pred,probas,testset_lbl,model_dir)
    print(confusion_matrix(y,y_pred))
    if len(X_arr)!=0: 
        metr=mtr.df_metrics(model,X_arr,y,y_pred,testset_lbl,y_probas=probas)
    else:
        metr=mtr.df_metrics(model,X,y,y_pred,testset_lbl,y_probas=probas)
    metrics_results.append(metr)
    print(classification_report(y,y_pred,digits=3))
    writeclassreporta(y,y_pred,"Dtestset "+testset_lbl[2:]+" \n", model_dir+'/classreport.txt')
    
def test_tags_thresholds(pdbsets):
    test_tags=["X_72","X_164","X_186","X_355","X_448"] if pdbsets==True else ["X_355","X_448"]
    tsets_thrshlds=[0.93,0.93,0.93,0.93,0.93] if pdbsets==True else [0.93,0.93]
    tsets_thrshlds_dict={}
    for i,test_tag in enumerate(test_tags):
        tsets_thrshlds_dict[test_tag]=tsets_thrshlds[i]
    return test_tags,tsets_thrshlds_dict

def plot_probas_valid(probas,model_dir):
    # print(min(probas),max(probas))
    # with open(model_dir+"/probas_valid.txt", 'w') as f:
    #     f.write("Min - max output: "+str(min(probas))+" "+str(max(probas))+ ' \n')
    #     f.close()
    count, bins, ignored = plt.hist(probas, 100)
    plt.title('Distribution of Predictions')
    plt.xlabel('Probability (predicted)')
    plt.ylabel('Count')
    plt.savefig(model_dir+'/prob_count_valid.jpg')
    plt.show()
    plt.close()

def plot_distrb_pred_valid(probas,y,model_dir):
    labels=["non-interaction","interaction"]
    #Distributions of Predicted Probabilities of both classes
    df = pd.DataFrame({'probas':probas, 'y': y})
    count, bins, ignored = plt.hist(df[df.y==0].probas, density=True, bins=100,
             alpha=.5, color='green',  label=labels[0])
    count, bins, ignored = plt.hist(df[df.y==1].probas, density=True, bins=100,
             alpha=.5, color='red', label=labels[1])
    # plt.axvline(.5, color='blue', linestyle='--', label='Boundary')
    # plt.xlim([0,1])
    plt.title('Distribution of Predictions')
    plt.xlabel('Probability (predicted)')
    plt.ylabel('Density')
    plt.legend(loc="upper right")
    plt.savefig(model_dir+'/prob_dist_valid.jpg')
    plt.show()
    plt.close()
    
def plot_probas(probas,model_dir,testset_lbl):
    # print(min(probas),max(probas))
    # with open(model_dir+"/probas_"+testset_lbl+".txt", 'w') as f:
    #     f.write("Min - max output: "+str(min(probas))+" "+str(max(probas))+ ' \n')
    #     f.close()
    count, bins, ignored = plt.hist(probas, 100)
    plt.title('Distribution of Predictions')
    plt.xlabel('Probability (predicted)')
    plt.ylabel('Count')
    plt.savefig(model_dir+'/prob_count_'+testset_lbl+'.jpg')
    plt.show()
    plt.close()

def plot_distrb_pred(probas,y,model_dir,testset_lbl):
    labels=["non-interaction","interaction"]
    #Distributions of Predicted Probabilities of both classes
    try:
        df = pd.DataFrame({'probas':probas, 'y': y})
    except:
        df = pd.DataFrame({'probas':probas[:,0], 'y': y[:]})
    count, bins, ignored = plt.hist(df[df.y==0].probas, density=True, bins=100,
             alpha=.5, color='green',  label=labels[0])
    count, bins, ignored = plt.hist(df[df.y==1].probas, density=True, bins=100,
             alpha=.5, color='red', label=labels[1])
    # plt.axvline(.5, color='blue', linestyle='--', label='Boundary')
    # plt.xlim([0,1])
    plt.title('Distribution of Predictions')
    plt.xlabel('Probability (predicted)')
    plt.ylabel('Density')
    plt.legend(loc="upper right")
    plt.savefig(model_dir+'/prob_dist_'+testset_lbl+'.jpg')
    plt.show()
    plt.close()

def testset_prediction_eval(model,X,y,model_dir,metrics_results,testset_lbl,
                            scale_data,scaler,clsf_model,threshold=0.5):
    probas = model.predict(X).ravel()
    plot_probas(probas,model_dir,testset_lbl)
    plot_distrb_pred(probas,y,model_dir,testset_lbl)
    labels = (probas >= threshold).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    mtr.plot_metrics_keras(y,y_pred,probas,testset_lbl,model_dir)
    print(confusion_matrix(y,y_pred))
    metr=mtr.df_metrics(model,X,y,y_pred,testset_lbl,y_probas=probas)
    metrics_results.append(metr)
    print(classification_report(y,y_pred,digits=3))
    writeclassreporta(y,y_pred,"Dtestset "+testset_lbl[2:]+" \n", model_dir+'/classreport.txt')
    del labels, X, y ,probas
    gc.collect()
    
def testset_prediction_evalML(model,X,y,model_dir,metrics_results,testset_lbl,
                            scale_data,scaler,clsf_model,threshold=0.5):
    probas = model.predict_proba(X)
    probas = np.delete(probas, 0, 1)
    plot_probas(probas,model_dir,testset_lbl)
    plot_distrb_pred(probas,y,model_dir,testset_lbl)
    labels = (probas >= threshold).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    mtr.plot_metrics_keras(y,y_pred,probas,testset_lbl,model_dir)
    print(confusion_matrix(y,y_pred))
    metr=mtr.df_metrics(model,X,y,y_pred,testset_lbl,y_probas=probas)
    metrics_results.append(metr)
    print(classification_report(y,y_pred,digits=3))
    writeclassreporta(y,y_pred,"Dtestset "+testset_lbl[2:]+" \n", model_dir+'/classreport.txt')
    del labels, X, y ,probas
    gc.collect()
    
def testset_prediction_eval_old(model,X,y,X_arr,model_dir,metrics_results,testset_lbl,scale_data,scaler,clsf_model):
    if scale_data==True:
        for i in range(len(X)):
            X[i]=scaler.fit_transform(X[i])
        if clsf_model=="CNN" and len(X.shape)!=4: X = np.reshape(X, (X.shape[0],X.shape[1],X.shape[2],1))
    if len(X_arr)!=0:
        if scale_data==True:
            for i in range(len(X_arr)):
                for j in range(len(X_arr[i])):
                    X_arr[i][j]=scaler.transform(X_arr[i][j])
            if len(X[1].shape)!=4:
                X_arr[1] = np.reshape(X_arr[1], (X_arr[1].shape[0],X_arr[1].shape[1],X_arr[1].shape[2],1))
        probas = model.predict(X_arr).ravel()
    else:
        probas = model.predict(X).ravel()
    labels = (probas >= 0.5).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    mtr.plot_metrics_keras(y,y_pred,probas,testset_lbl,model_dir)
    print(confusion_matrix(y,y_pred))
    if len(X_arr)!=0: 
        metr=mtr.df_metrics(model,X_arr,y,y_pred,testset_lbl,y_probas=probas)
    else:
        metr=mtr.df_metrics(model,X,y,y_pred,testset_lbl,y_probas=probas)
    metrics_results.append(metr)
    print(classification_report(y,y_pred,digits=3))
    writeclassreporta(y,y_pred,"Dtestset "+testset_lbl[2:]+" \n", model_dir+'/classreport.txt')
    del labels, X, y, X_arr,probas
    gc.collect()

def testset_prediction_eval2(model,X,y,X_arr,model_dir,metrics_results,testset_lbl,thrs):
    if len(X_arr)!=0:
        probas = model.predict(X_arr).ravel()
    else:
        probas = model.predict(X).ravel()
    labels = (probas >= thrs).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    mtr.plot_metrics_keras(y,y_pred,probas,testset_lbl,model_dir)
    print(confusion_matrix(y,y_pred))
    metr=mtr.df_metrics(model,X,y,y_pred,testset_lbl,y_probas=probas)
    metrics_results.append(metr)
    print(classification_report(y,y_pred,digits=3))
    writeclassreporta(y,y_pred,"Dtestset "+testset_lbl[2:]+" \n", model_dir+'/classreport.txt')
    
def testset_prediction_eval_thrs(model,X,y,X_arr,model_dir,metrics_results,testset_lbl,thrs):
    if len(X_arr)!=0:
        probas = model.predict(X_arr).ravel()
    else:
        probas = model.predict(X).ravel()
    labels = (probas >= thrs).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    print(confusion_matrix(y,y_pred))
    metr=mtr.df_metrics(model,X,y,y_pred,testset_lbl,y_probas=probas)
    metrics_results.append(metr)
    print(classification_report(y,y_pred,digits=3))
    writeclassreporta(y,y_pred,"Dtestset "+testset_lbl[2:]+" \n", model_dir+'/classreport_thrs.txt')
    
def RNN(results_dir):
    if undersampling==True:
        under_1_1_training=True
        under_1_1_valid=True
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl="_down"
    else:
        under_1_1_training=False
        under_1_1_valid=False
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl=""
    training_dataset="df_alldataW_all.csv"
    X_train, y_train, X_valid, y_valid, \
    X_355, y_355, \
    X_448, y_448 = load_data(under_lbl)
    
    tf.keras.backend.clear_session()
    optimizer_name='adam'
    learning_rate=0.002
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss='binary_crossentropy'
    metrics=[tf.keras.metrics.AUC(curve='PR')]
    patience=4
    min_delta=0.001
    restore_best_weights=True
    validation_split=0.2
    batch_size=1024
    steps_per_epoch=500
    epochs=9
    padding='same'
    strides=(1, 1) 
    kernel_size=5
    n_filter=48
    max_pool=(2,2)
    gru_unit=64
    
    print(X_train.shape)
    early_stopping = keras.callbacks.EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights
    )
    callbacks=[early_stopping]
    w0=X_train.shape[0]/(2*(X_train.shape[0]-np.sum(y_train,dtype=int)))
    w1=X_train.shape[0]/(2*np.sum(y_train,dtype=int))
    if sqrtw==True:
    	w0=math.sqrt(w0)
    	w1=math.sqrt(w1)
    if classwcomp==True: 
        class_weight={0: w0, 1: w1}
    else:
        class_weight={0: 0.55, 1: 4.97}
    print(class_weight)
    if sampw==True:
        sample_weight=np.array([w0 if i==0 else w1 for i in y_train.tolist()])
    else:
        sample_weight=None
    shuffle=True
    print(X_train.shape[0],X_train.shape[1],X_train.shape[2])
    start_time = time.time()
    start = time.process_time()
    model = models.Sequential()
    model.add(layers.Bidirectional(layers.GRU(gru_unit), input_shape=X_train.shape[1:]))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.summary()
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    if tf.test.is_gpu_available():
        print("GPUs available: ")
        print(tf.config.experimental.list_physical_devices('XLA_GPU'))
        print(tf.config.experimental.list_physical_devices('GPU'))
    print("Devices List: ")
    print(device_lib.list_local_devices())
    if device_type=="GPU":
        device_t="/device:GPU:0"
    else:
        device_t="/device:CPU:0"
        
    if classw==True or classwcomp==True:
        class_weight=class_weight
    else: 
        class_weight=None
    with tf.device(device_t):
        history = model.fit(
            X_train, y_train, 
            epochs=epochs, 
            validation_data=(X_valid, y_valid),
            callbacks=callbacks,
            batch_size=batch_size,
            class_weight=class_weight,
            sample_weight=sample_weight,
            shuffle=shuffle
        )
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    
    print("process time: ", ptime, "real time: ", rtime)
    
    # timestr = time.strftime("%Y%m%d_%H%M%S")
    # print("timestr: ", timestr)
    # results_dir="./results_rnn"
    # if not os.path.exists(results_dir): os.mkdir(results_dir)
    # timestr='20210508_150602'
    # model = keras.models.load_model(results_dir+'/'+'deeplearning3tW_d1_model_20210508_150602')
    # history_df=pd.read_csv(model_dir+'/'+'deeplearning3tW_d1_model_20210508_150602_history.csv', index_col=0, header=0)
    model_dir=results_dir+'/model_rnn'
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model.save(model_dir+'/model.h5')
    model.save_weights(model_dir+"/weights.h5")
    orig_stdout = sys.stdout
    f = open(model_dir+"/devices.txt", 'w+')
    sys.stdout = f
    for device in device_lib.list_local_devices():
        print(device)
    sys.stdout = orig_stdout
    f.close()
    mparams_file=model_dir+"/modelparams.txt"
    with open(mparams_file, 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.write("device used: "+device_t+ '\n')
        f.write("optimizer: "+optimizer_name+ '\n')
        f.write("loss: "+loss+ '\n')
        f.write("metrics: "+str(metrics)+ '\n')
        f.write("patience: "+ str(patience)+ '\n')
        f.write("min_delta: "+str(min_delta)+ '\n')
        f.write("restore_best_weights: "+str(restore_best_weights)+ '\n')
        f.write("gru_unit: "+str(gru_unit)+ '\n')
        # f.write("validation_split: "+str(validation_split)+ '\n')
        # f.write("batch_size: "+str(batch_size)+ '\n')
        # f.write("steps_per_epoch: "+str(steps_per_epoch)+ '\n')
        # f.write("epochs: "+str(epochs)+ '\n')
        # f.write("padding: "+padding+ '\n')
        # f.write("strides: "+str(strides)+ '\n')
        # f.write("kernel_size: "+str(kernel_size)+ '\n')
        # f.write("n_filter: "+str(n_filter)+ '\n')
        # f.write("max_pool: "+str(max_pool)+ '\n')
        f.write("callbacks: "+"[early_stopping]"+ '\n')
        #f.write("class_weight: "+str(class_weight)+ '\n')
        f.write("shuffle: "+str(shuffle)+ '\n')
        f.write("Arguments: "+str(vars(args))+ '\n')
        f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ '\n')
        f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ '\n')
        f.write("Test dataset 355 undersampled (1:1): "+str(under_1_1_test_355)+ '\n')
        f.write("Test dataset 448 undersampled (1:1): "+str(under_1_1_test_448)+ '\n')
        f.write("Training dataset: "+training_dataset+ '\n')
        # f.write("The length of training dataset: "+str(len(df1))+ '\n')
        # f.write("The length of case study(cs) dataset: "+str(len(df_cs))+ '\n')
        # f.write("The length of training dataset w/o cs: "+str(len(df))+ '\n')
        f.write("The number of columns of training dataset w/o label: "+str(X_train.shape[1])+ '\n')
        f.write("The size of sliding windows : "+str(X_train.shape[2])+ '\n')
        f.write("The length of training dataset after splitting: "+str(X_train.shape[0])+ '\n')
        f.write("The length of validation dataset: "+str(len(X_valid))+ '\n')
        f.close()
        
    plot_model_history(history,model_dir)
    
    metrics_results=[]
    scaler="Standart"
    clsf_model="CNN"
    print("Dtestset 355 \n")
    test_dataset_355="df_alldataW_Dset_355.csv"
    testset_prediction_eval(model,X_355,y_355,[],model_dir,metrics_results,"X_355",scale_data,scaler,clsf_model)
    
    print("Dtestset 448 \n")
    test_dataset_448="df_alldataW_Dset_448.csv"
    testset_prediction_eval(model,X_448,y_448,[],model_dir,metrics_results,"X_448",scale_data,scaler,clsf_model)
    
    with open(mparams_file, 'a') as f:
        f.write("Test dataset: "+test_dataset_355+ '\n')
        f.write("The length of test dataset 355: "+str(len(X_355))+ '\n')
        f.write("Test dataset: "+test_dataset_448+ '\n')
        f.write("The length of test dataset 448: "+str(len(X_448))+ '\n')
        f.close()
    df_metrics_all=pd.concat(metrics_results)
    print(df_metrics_all)
    
    df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
    df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")

def CNN(results_dir):
    if undersampling==True:
        under_1_1_training=True
        under_1_1_valid=True
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl="_down"
    else:
        under_1_1_training=False
        under_1_1_valid=False
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl=""
    training_dataset="df_alldataW_all.csv"
    X_train, y_train, X_valid, y_valid, \
    X_355, y_355, \
    X_448, y_448 = load_data(under_lbl)
    
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
    X_valid = np.reshape(X_valid, (X_valid.shape[0],X_valid.shape[1],X_valid.shape[2],1))
    X_355 = np.reshape(X_355, (X_355.shape[0],X_355.shape[1],X_355.shape[2],1))
    X_448 = np.reshape(X_448, (X_448.shape[0],X_448.shape[1],X_448.shape[2],1))
    
    tf.keras.backend.clear_session()
    optimizer_name='adam'
    learning_rate=0.002
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss='binary_crossentropy'
    metrics=[tf.keras.metrics.AUC(curve='PR')]
    patience=4
    min_delta=0.001
    restore_best_weights=True
    validation_split=0.2
    batch_size=1024
    steps_per_epoch=500
    epochs=8
    padding='same'
    strides=(1, 1) 
    kernel_size=5
    n_filter=48
    max_pool=(2,2)
    input_shape=X_train.shape[1:]
    early_stopping = keras.callbacks.EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights
    )
    callbacks=[early_stopping]
    print(X_train.shape)
    w0=X_train.shape[0]/(2*(X_train.shape[0]-np.sum(y_train,dtype=int)))
    print(X_train.shape[0],np.sum(y_train,dtype=int))
    w1=X_train.shape[0]/(2*np.sum(y_train,dtype=int))
    class_weight={0: w0, 1: w1}
    #class_weight={0: 0.55, 1: 4.97}
    if sqrtw==True:
    	w0=math.sqrt(w0)
    	w1=math.sqrt(w1)
    if classwcomp==True: 
        class_weight={0: w0, 1: w1}
    else:
        class_weight={0: 0.55, 1: 4.97}
    print(class_weight)
    if sampw==True:
        sample_weight=np.array([w0 if i==0 else w1 for i in y_train.tolist()])
    else:
        sample_weight=None
    shuffle=True
    start_time = time.time()
    start = time.process_time()
    model = models.Sequential()
    model.add(layers.Conv2D(n_filter, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(max_pool,padding=padding))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.summary()
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    if tf.test.is_gpu_available():
        print("GPUs available: ")
        print(tf.config.experimental.list_physical_devices('XLA_GPU'))
        print(tf.config.experimental.list_physical_devices('GPU'))
    print("Devices List: ")
    print(device_lib.list_local_devices())
    if device_type=="GPU":
        device_t="/device:GPU:0"
    else:
        device_t="/device:CPU:0"
        
    if classw==True or classwcomp==True:
        class_weight=class_weight
    else: 
        class_weight=None
    with tf.device(device_t):
        history = model.fit(
            X_train, y_train, 
            epochs=epochs, 
            validation_data=(X_valid, y_valid),
            callbacks=callbacks,
            batch_size=batch_size,
            class_weight=class_weight,
            sample_weight=sample_weight,
            shuffle=shuffle
        )
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    
    print("process time: ", ptime, "real time: ", rtime)
    
    # timestr = time.strftime("%Y%m%d_%H%M%S")
    # print("timestr: ", timestr)
    # results_dir="./results_cnn"
    # if not os.path.exists(results_dir): os.mkdir(results_dir)
    # timestr='20210508_150602'
    # model = keras.models.load_model(results_dir+'/'+'deeplearning3tW_d1_model_20210508_150602')
    # history_df=pd.read_csv(model_dir+'/'+'deeplearning3tW_d1_model_20210508_150602_history.csv', index_col=0, header=0)
    model_dir=results_dir+'/model_cnn'
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model.save(model_dir+'/model.h5')
    model.save_weights(model_dir+"/weights.h5")
    orig_stdout = sys.stdout
    f = open(model_dir+"/devices.txt", 'w+')
    sys.stdout = f
    for device in device_lib.list_local_devices():
        print(device)
    sys.stdout = orig_stdout
    f.close()
    mparams_file=model_dir+"/modelparams.txt"
    with open(mparams_file, 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.write("program name: "+sys.argv[0]+ '\n')
        f.write("device used: "+device_t+ '\n')
        f.write("optimizer: "+optimizer_name+ '\n')
        f.write("loss: "+loss+ '\n')
        f.write("metrics: "+ str(metrics)+ '\n')
        f.write("patience: "+ str(patience)+ '\n')
        f.write("min_delta: "+str(min_delta)+ '\n')
        f.write("restore_best_weights: "+str(restore_best_weights)+ '\n')
        # f.write("validation_split: "+str(validation_split)+ '\n')
        f.write("batch_size: "+str(batch_size)+ '\n')
        # f.write("steps_per_epoch: "+str(steps_per_epoch)+ '\n')
        f.write("epochs: "+str(epochs)+ '\n')
        f.write("padding: "+padding+ '\n')
        f.write("strides: "+str(strides)+ '\n')
        f.write("kernel_size: "+str(kernel_size)+ '\n')
        f.write("n_filter: "+str(n_filter)+ '\n')
        f.write("max_pool: "+str(max_pool)+ '\n')
        f.write("callbacks: "+"[early_stopping]"+ '\n')
        #f.write("class_weight: "+str(class_weight)+ '\n')
        f.write("shuffle: "+str(shuffle)+ '\n')
        f.write("Arguments: "+str(vars(args))+ '\n')
        f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ '\n')
        f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ '\n')
        f.write("Test dataset 355 undersampled (1:1): "+str(under_1_1_test_355)+ '\n')
        f.write("Test dataset 448 undersampled (1:1): "+str(under_1_1_test_448)+ '\n')
        f.write("Training dataset: "+training_dataset+ '\n')
        # f.write("The length of training dataset: "+str(len(df1))+ '\n')
        # f.write("The length of case study(cs) dataset: "+str(len(df_cs))+ '\n')
        # f.write("The length of training dataset w/o cs: "+str(len(df))+ '\n')
        f.write("The number of columns of training dataset w/o label: "+str(X_train.shape[1])+ '\n')
        f.write("The size of sliding windows : "+str(X_train.shape[2])+ '\n')
        f.write("The length of training dataset after splitting: "+str(X_train.shape[0])+ '\n')
        f.write("The length of validation dataset: "+str(len(X_valid))+ '\n')
        f.close()
        
    plot_model_history(history,model_dir)
    
    metrics_results=[]
    scaler="Standart"
    clsf_model="CNN"
    print("Dtestset 355 \n")
    test_dataset_355="df_alldataW_Dset_355.csv"
    testset_prediction_eval(model,X_355,y_355,[],model_dir,metrics_results,"X_355",scale_data,scaler,clsf_model)
    
    print("Dtestset 448 \n")
    test_dataset_448="df_alldataW_Dset_448.csv"
    testset_prediction_eval(model,X_448,y_448,[],model_dir,metrics_results,"X_448",scale_data,scaler,clsf_model)
    
    with open(mparams_file, 'a') as f:
        f.write("Test dataset: "+test_dataset_355+ '\n')
        f.write("The length of test dataset 355: "+str(len(X_355))+ '\n')
        f.write("Test dataset: "+test_dataset_448+ '\n')
        f.write("The length of test dataset 448: "+str(len(X_448))+ '\n')
        f.close()
    df_metrics_all=pd.concat(metrics_results)
    print(df_metrics_all)
    
    df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
    df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
    
def Ensemble(results_dir):
    if undersampling==True:
        under_1_1_training=True
        under_1_1_valid=True
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl="_under"
    else:
        under_1_1_training=False
        under_1_1_valid=False
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl=""
    training_dataset="df_alldataW_all.csv"
    X_train, y_train, X_valid, y_valid, \
    X_72, y_72, X_164, y_164, X_186, y_186, X_355, y_355, \
    X_448, y_448 = load_data(under_lbl)
    X_train_r1 = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
    X_valid_r1 = np.reshape(X_valid, (X_valid.shape[0],X_valid.shape[1],X_valid.shape[2],1))
    X_355_r1 = np.reshape(X_355, (X_355.shape[0],X_355.shape[1],X_355.shape[2],1))
    X_448_r1 = np.reshape(X_448, (X_448.shape[0],X_448.shape[1],X_448.shape[2],1))
    
    tf.keras.backend.clear_session()
    optimizer_name='adam'
    learning_rate=0.002
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss='binary_crossentropy'
    metrics=[tf.keras.metrics.AUC(curve='PR')]
    patience=4
    min_delta=0.001
    restore_best_weights=True
    validation_split=0.2
    batch_size=1024
    steps_per_epoch=500
    epochs=5
    padding='same'
    strides=(1, 1) 
    kernel_size=5
    n_filter=48
    max_pool=(2,2)
    gru_unit=64
    input_shape=X_train.shape[1:]
    early_stopping = keras.callbacks.EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights
    )
    callbacks=[early_stopping]
    w0=X_train.shape[0]/(2*(X_train.shape[0]-np.sum(y_train,dtype=int)))
    w1=X_train.shape[0]/(2*np.sum(y_train,dtype=int))
    class_weight={0: w0, 1: w1}
    #class_weight={0: 0.55, 1: 4.97}
    if sqrtw==True:
    	w0=math.sqrt(w0)
    	w1=math.sqrt(w1)
    if classwcomp==True: 
        class_weight={0: w0, 1: w1}
    else:
        class_weight={0: 0.55, 1: 4.97}
    print(class_weight)
    if sampw==True:
        sample_weight=np.array([w0 if i==0 else w1 for i in y_train.tolist()])
    else:
        sample_weight=None
    shuffle=True
    print(X_train.shape[0],X_train.shape[1],X_train.shape[2])
    # res_dir=sys.argv[2]
    # results_dir=res_dir+"dplrnn1/results"
    model_folder_rnn=results_dir+'/model_rnn/model.h5'
    model3 = keras.models.load_model(model_folder_rnn)
    # results_dir=res_dir+"dplcnn1/results"
    model_folder_cnn=results_dir+'/model_cnn/model.h5'
    model4 = keras.models.load_model(model_folder_cnn)
    layer1_1=model3.get_layer(index=0)
    layer1_1.trainable = False
    layer1_2=model3.get_layer(index=1)
    layer1_2.trainable = False
    layer2_1=model4.get_layer(index=0)
    layer2_1.trainable = False
    layer2_2=model4.get_layer(index=1)
    layer2_2.trainable = False
    layer2_3=model4.get_layer(index=2)
    layer2_3.trainable = False
    start_time = time.time()
    start = time.process_time()
    modela = Sequential()
    modela.add(layer1_1)
    modela.add(layers.Flatten())
    modelb = Sequential()
    modelb.add(layer2_1)
    modelb.add(layer2_2)
    modelb.add(layers.Flatten())
    
    merged_output = concatenate([modela.output, modelb.output])   
    
    model_combined = Sequential()
    model_combined.add(layers.Dense(96, activation='relu'))
    model_combined.add(layers.Dropout(0.3))
    model_combined.add(layers.Dense(1,activation='sigmoid'))
    
    model = Model([modela.input, modelb.input], model_combined(merged_output))
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    if tf.test.is_gpu_available():
        print("GPUs available: ")
        print(tf.config.experimental.list_physical_devices('XLA_GPU'))
        print(tf.config.experimental.list_physical_devices('GPU'))
    print("Devices List: ")
    print(device_lib.list_local_devices())
    if device_type=="GPU":
        device_t="/device:GPU:0"
    else:
        device_t="/device:CPU:0"
        
    if classw==True or classwcomp==True:
        class_weight=class_weight
    else: 
        class_weight=None
    with tf.device(device_t):
        history = model.fit(
            x=[X_train,X_train_r1], y=y_train, 
            epochs=epochs, 
            validation_data=([X_valid,X_valid_r1],y_valid),
            callbacks=callbacks,
            batch_size=batch_size,
            class_weight=class_weight,
            sample_weight=sample_weight,
            shuffle=shuffle
        )
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    
    print("process time: ", ptime, "real time: ", rtime)
    
    # timestr = time.strftime("%Y%m%d_%H%M%S")
    # print("timestr: ", timestr)
    # results_dir="./results"
    # if not os.path.exists(results_dir): os.mkdir(results_dir)
    # timestr='20210508_150602'
    # model = keras.models.load_model(results_dir+'/'+'deeplearning3tW_d1_model_20210508_150602')
    # history_df=pd.read_csv(model_dir+'/'+'deeplearning3tW_d1_model_20210508_150602_history.csv', index_col=0, header=0)
    model_dir=results_dir+'/model_ensb'
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model.save(model_dir+"/model.h5")
    model.save_weights(model_dir+"/weights.h5")
    orig_stdout = sys.stdout
    f = open(model_dir+"/devices.txt", 'w+')
    sys.stdout = f
    for device in device_lib.list_local_devices():
        print(device)
    sys.stdout = orig_stdout
    f.close()
    mparams_file=model_dir+"/modelparams.txt"
    with open(mparams_file, 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.write("program name: "+sys.argv[0]+ '\n')
        f.write("device used: "+device_t+ '\n')
        f.write("model rnn: "+model_folder_rnn+ '\n')
        f.write("model cnn: "+model_folder_cnn+ '\n')
        f.write("optimizer: "+optimizer_name+ '\n')
        f.write("loss: "+loss+ '\n')
        f.write("metrics: "+str(metrics)+ '\n')
        f.write("patience: "+ str(patience)+ '\n')
        f.write("min_delta: "+str(min_delta)+ '\n')
        f.write("restore_best_weights: "+str(restore_best_weights)+ '\n')
        f.write("gru_unit: "+str(gru_unit)+ '\n')
        # f.write("validation_split: "+str(validation_split)+ '\n')
        # f.write("batch_size: "+str(batch_size)+ '\n')
        # f.write("steps_per_epoch: "+str(steps_per_epoch)+ '\n')
        # f.write("epochs: "+str(epochs)+ '\n')
        # f.write("padding: "+padding+ '\n')
        # f.write("strides: "+str(strides)+ '\n')
        # f.write("kernel_size: "+str(kernel_size)+ '\n')
        # f.write("n_filter: "+str(n_filter)+ '\n')
        # f.write("max_pool: "+str(max_pool)+ '\n')
        f.write("callbacks: "+"[early_stopping]"+ '\n')
        #f.write("class_weight: "+str(class_weight)+ '\n')
        f.write("shuffle: "+str(shuffle)+ '\n')
        f.write("Arguments: "+str(vars(args))+ '\n')
        f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ '\n')
        f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ '\n')
        f.write("Test dataset 355 undersampled (1:1): "+str(under_1_1_test_355)+ '\n')
        f.write("Test dataset 448 undersampled (1:1): "+str(under_1_1_test_448)+ '\n')
        f.write("Training dataset: "+training_dataset+ '\n')
        # f.write("The length of training dataset: "+str(len(df1))+ '\n')
        # f.write("The length of case study(cs) dataset: "+str(len(df_cs))+ '\n')
        # f.write("The length of training dataset w/o cs: "+str(len(df))+ '\n')
        f.write("The number of columns of training dataset w/o label: "+str(len(X_train[0][0]))+ '\n')
        f.write("The size of sliding windows : "+str(len(X_train[0]))+ '\n')
        f.write("The length of training dataset after splitting: "+str(len(X_train))+ '\n')
        f.write("The length of validation dataset: "+str(len(X_valid))+ '\n')
        f.close()
        
    plot_model_history(history,model_dir)
    
    metrics_results=[]
    
    print("Dtestset 355 \n")
    test_dataset_355="df_alldataW_Dset_355.csv"
    testset_prediction_eval(model,X_355,y_355,[X_355,X_355_r1],model_dir,metrics_results,"X_355")
    
    print("Dtestset 448 \n")
    test_dataset_448="df_alldataW_Dset_448.csv"
    testset_prediction_eval(model,X_448,y_448,[X_448,X_448_r1],model_dir,metrics_results,"X_448")
    
    with open(mparams_file, 'a') as f:
        f.write("Test dataset: "+test_dataset_355+ '\n')
        f.write("The length of test dataset 355: "+str(len(X_355))+ '\n')
        f.write("Test dataset: "+test_dataset_448+ '\n')
        f.write("The length of test dataset 448: "+str(len(X_448))+ '\n')
        f.close()
    df_metrics_all=pd.concat(metrics_results)
    print(df_metrics_all)
    
    df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
    df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")

def stacked_generalization(results_dir):
    if undersampling==True:
        under_1_1_training=True
        under_1_1_valid=True
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl="_under"
    else:
        under_1_1_training=False
        under_1_1_valid=False
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl=""
    training_dataset="df_alldataW_all.csv"
    X_train, y_train, X_valid, y_valid, \
    X_72, y_72, X_164, y_164, X_186, y_186, X_355, y_355, \
    X_448, y_448 = load_data(under_lbl)
    X_train_r1 = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
    X_valid_r1 = np.reshape(X_valid, (X_valid.shape[0],X_valid.shape[1],X_valid.shape[2],1))
    X_355_r1 = np.reshape(X_355, (X_355.shape[0],X_355.shape[1],X_355.shape[2],1))
    X_448_r1 = np.reshape(X_448, (X_448.shape[0],X_448.shape[1],X_448.shape[2],1))
    
    tf.keras.backend.clear_session()
    optimizer_name='adam'
    learning_rate=0.002
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss='binary_crossentropy'
    metrics=[tf.keras.metrics.AUC(curve='PR')]
    patience=4
    min_delta=0.001
    restore_best_weights=True
    validation_split=0.2
    batch_size=1024
    steps_per_epoch=500
    epochs=5
    padding='same'
    strides=(1, 1) 
    kernel_size=5
    n_filter=48
    max_pool=(2,2)
    dense_units=96
    dropout_rate=0.3
    input_shape=X_train.shape[1:]
    early_stopping = keras.callbacks.EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights
    )
    callbacks=[early_stopping]
    w0=X_train.shape[0]/(2*(X_train.shape[0]-np.sum(y_train,dtype=int)))
    w1=X_train.shape[0]/(2*np.sum(y_train,dtype=int))
    class_weight={0: w0, 1: w1}
    #class_weight={0: 0.55, 1: 4.97}
    if sqrtw==True:
    	w0=math.sqrt(w0)
    	w1=math.sqrt(w1)
    if classwcomp==True: 
        class_weight={0: w0, 1: w1}
    else:
        class_weight={0: 0.55, 1: 4.97}
    print(class_weight)
    if sampw==True:
        sample_weight=np.array([w0 if i==0 else w1 for i in y_train.tolist()])
    else:
        sample_weight=None
    shuffle=True
    print(X_train.shape[0],X_train.shape[1],X_train.shape[2])
    # res_dir=sys.argv[2]
    # results_dir=res_dir+"dplrnn1/results"
    n_models_rnn=5
    n_models_cnn=5
    model = build_ensemble_model(results_dir,n_models_rnn,n_models_cnn,dense_units,
                             dropout_rate,optimizer,loss,metrics)
    model.summary()
    # plot graph of ensemble
    plot_model_history(model, show_shapes=True, to_file=results_dir+'/model_graph.png')
    
    start_time = time.time()
    start = time.process_time()
    if tf.test.is_gpu_available():
        print("GPUs available: ")
        print(tf.config.experimental.list_physical_devices('XLA_GPU'))
        print(tf.config.experimental.list_physical_devices('GPU'))
    print("Devices List: ")
    print(device_lib.list_local_devices())
    if device_type=="GPU":
        device_t="/device:GPU:0"
    else:
        device_t="/device:CPU:0"
        
    if classw==True or classwcomp==True:
        class_weight=class_weight
    else: 
        class_weight=None
    with tf.device(device_t):
        history = model.fit(
            x=[X_train,X_train_r1], y=y_train, 
            epochs=epochs, 
            validation_data=([X_valid,X_valid_r1],y_valid),
            callbacks=callbacks,
            batch_size=batch_size,
            class_weight=class_weight,
            sample_weight=sample_weight,
            shuffle=shuffle
        )
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    
    print("process time: ", ptime, "real time: ", rtime)
    
    # timestr = time.strftime("%Y%m%d_%H%M%S")
    # print("timestr: ", timestr)
    # results_dir="./results"
    # if not os.path.exists(results_dir): os.mkdir(results_dir)
    # timestr='20210508_150602'
    # model = keras.models.load_model(results_dir+'/'+'deeplearning3tW_d1_model_20210508_150602')
    # history_df=pd.read_csv(model_dir+'/'+'deeplearning3tW_d1_model_20210508_150602_history.csv', index_col=0, header=0)
    model_dir=results_dir+'/model_ensb'
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model.save(model_dir+"/model.h5")
    model.save_weights(model_dir+"/weights.h5")
    orig_stdout = sys.stdout
    f = open(model_dir+"/devices.txt", 'w+')
    sys.stdout = f
    for device in device_lib.list_local_devices():
        print(device)
    sys.stdout = orig_stdout
    f.close()
    mparams_file=model_dir+"/modelparams.txt"
    with open(mparams_file, 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.write("program name: "+sys.argv[0]+ '\n')
        f.write("device used: "+device_t+ '\n')
        # f.write("model rnn: "+model_folder_rnn+ '\n')
        # f.write("model cnn: "+model_folder_cnn+ '\n')
        f.write("optimizer: "+optimizer_name+ '\n')
        f.write("loss: "+loss+ '\n')
        f.write("metrics: "+str(metrics)+ '\n')
        f.write("patience: "+ str(patience)+ '\n')
        f.write("min_delta: "+str(min_delta)+ '\n')
        f.write("restore_best_weights: "+str(restore_best_weights)+ '\n')
        # f.write("validation_split: "+str(validation_split)+ '\n')
        # f.write("batch_size: "+str(batch_size)+ '\n')
        # f.write("steps_per_epoch: "+str(steps_per_epoch)+ '\n')
        # f.write("epochs: "+str(epochs)+ '\n')
        # f.write("padding: "+padding+ '\n')
        # f.write("strides: "+str(strides)+ '\n')
        # f.write("kernel_size: "+str(kernel_size)+ '\n')
        # f.write("n_filter: "+str(n_filter)+ '\n')
        # f.write("max_pool: "+str(max_pool)+ '\n')
        f.write("callbacks: "+"[early_stopping]"+ '\n')
        #f.write("class_weight: "+str(class_weight)+ '\n')
        f.write("shuffle: "+str(shuffle)+ '\n')
        f.write("Arguments: "+str(vars(args))+ '\n')
        f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ '\n')
        f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ '\n')
        f.write("Test dataset 355 undersampled (1:1): "+str(under_1_1_test_355)+ '\n')
        f.write("Test dataset 448 undersampled (1:1): "+str(under_1_1_test_448)+ '\n')
        f.write("Training dataset: "+training_dataset+ '\n')
        # f.write("The length of training dataset: "+str(len(df1))+ '\n')
        # f.write("The length of case study(cs) dataset: "+str(len(df_cs))+ '\n')
        # f.write("The length of training dataset w/o cs: "+str(len(df))+ '\n')
        f.write("The number of columns of training dataset w/o label: "+str(len(X_train[0][0]))+ '\n')
        f.write("The size of sliding windows : "+str(len(X_train[0]))+ '\n')
        f.write("The length of training dataset after splitting: "+str(len(X_train))+ '\n')
        f.write("The length of validation dataset: "+str(len(X_valid))+ '\n')
        f.close()
        
    plot_model_history(history,model_dir)
    
    metrics_results=[]
    
    print("Dtestset 355 \n")
    test_dataset_355="df_alldataW_Dset_355.csv"
    testset_prediction_eval(model,X_355,y_355,[X_355,X_355_r1],model_dir,metrics_results,"X_355")
    
    print("Dtestset 448 \n")
    test_dataset_448="df_alldataW_Dset_448.csv"
    testset_prediction_eval(model,X_448,y_448,[X_448,X_448_r1],model_dir,metrics_results,"X_448")
    
    with open(mparams_file, 'a') as f:
        f.write("Test dataset: "+test_dataset_355+ '\n')
        f.write("The length of test dataset 355: "+str(len(X_355))+ '\n')
        f.write("Test dataset: "+test_dataset_448+ '\n')
        f.write("The length of test dataset 448: "+str(len(X_448))+ '\n')
        f.close()
    df_metrics_all=pd.concat(metrics_results)
    print(df_metrics_all)
    
    df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
    df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")

def scale_datasets(X,X_tests,scaler_type):
    if scaler_type=="Standard":
        scaler = StandardScaler()
    elif scaler_type=="MinMax":
        scaler = MinMaxScaler()
    elif scaler_type=="Normalizer":
        scaler = Normalizer(norm='max')
    for i in range(len(X)):
        X[i]=scaler.fit_transform(X[i])
    for X_test in X_tests:
        for i in range(len(X_test)):
            X_test[i]=scaler.transform(X_test[i])
    return X, X_tests, scaler

def scale_datasets2(X,X_tests,scaler_type):
    if scaler_type=="Standard":
        scaler = StandardScaler()
    elif scaler_type=="MinMax":
        scaler = MinMaxScaler()
    elif scaler_type=="Normalizer":
        scaler = Normalizer(norm='max')
    X=scaler.fit_transform(X)
    for X_test in X_tests:
        X_test=scaler.transform(X_test)
    return X, X_tests, scaler

def build_lstms2s_model_bnorm(input_shape, lstm_units, loss, optimizer, metrics, tag, fold_no):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=input_shape,name=tag+"_encs2sinput_"+str(fold_no))
    encoder = layers.LSTM(lstm_units, return_state=True, name=tag+"_encs2sLstm1_"+str(fold_no))
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    
    # Set up the decoder, using `encoder_states` as initial state.
    
    decoder_inputs = keras.Input(shape=(1,1),name=tag+"_decs2sinput_"+str(fold_no))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True,
                               name=tag+"_decs2sLstm1_"+str(fold_no))
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_bnorm = keras.layers.BatchNormalization(name=tag+"_s2sBatch1_"+str(fold_no))
    decoder_outputs = decoder_bnorm(decoder_outputs)
    rshape=layers.Reshape([lstm_units],name=tag+"_s2sreshape1_"+str(fold_no))
    decoder_outputs = rshape(decoder_outputs) 
    
    decoder_dense = keras.layers.Dense(1, activation="sigmoid",
                                       name=tag+"_s2sDense1_"+str(fold_no))
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_lstms2s_model_bnorm2(input_shape, lstm_units, loss, optimizer, metrics, tag, fold_no):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=input_shape,name=tag+"_encs2sinput_"+str(fold_no))
    encoder = layers.Bidirectional(layers.LSTM(lstm_units, return_state=True, name=tag+"_encs2sLstm1_"+str(fold_no)),
                                   name=tag+"_encBilstm1_"+str(fold_no))
    encoder_outputs, forw_state_h, forw_state_c, back_state_h, back_state_c = encoder(encoder_inputs)
    
    # We discard `encoder_outputs` and only keep the states.
    final_enc_h = layers.Concatenate(name=tag+"_encConcat1_"+str(fold_no))([forw_state_h,back_state_h])
    final_enc_c = layers.Concatenate(name=tag+"_encConcat2_"+str(fold_no))([forw_state_c,back_state_c])
    # Set up the decoder, using `encoder_states` as initial state.
    encoder_states =[final_enc_h, final_enc_c]
    
    decoder_inputs = keras.Input(shape=(1,1),name=tag+"_decs2sinput_"+str(fold_no))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = keras.layers.LSTM(2*lstm_units, return_sequences=True, return_state=True,
                               name=tag+"_decs2sLstm1_"+str(fold_no))
    decoder_outputs, _,_ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_bnorm = keras.layers.BatchNormalization(name=tag+"_s2sBatch1_"+str(fold_no))
    decoder_outputs = decoder_bnorm(decoder_outputs)
    rshape=layers.Reshape([2*lstm_units],name=tag+"_s2sreshape1_"+str(fold_no))
    decoder_outputs = rshape(decoder_outputs) 
    
    decoder_dense = keras.layers.Dense(1, activation="sigmoid",
                                       name=tag+"_s2sDense1_"+str(fold_no))
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_lstms2s_model_bnorm3(input_shape, lstm_units, loss, optimizer, metrics, tag, fold_no):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=input_shape,name=tag+"_encs2sinput_"+str(fold_no))
    encoder = layers.Bidirectional(layers.LSTM(lstm_units, return_state=True, name=tag+"_encs2sLstm1_"+str(fold_no)),
                                   name=tag+"_encBilstm1_"+str(fold_no))
    encoder_outputs, forw_state_h, forw_state_c, back_state_h, back_state_c = encoder(encoder_inputs)
    
    # We discard `encoder_outputs` and only keep the states.
    # final_enc_h = layers.Concatenate(name=tag+"_encConcat1_"+str(fold_no))([forw_state_h,back_state_h])
    # final_enc_c = layers.Concatenate(name=tag+"_encConcat2_"+str(fold_no))([forw_state_c,back_state_c])
    # Set up the decoder, using `encoder_states` as initial state.
    encoder_states =[forw_state_h, forw_state_c, back_state_h, back_state_c]
    
    decoder_inputs = keras.Input(shape=(1,1),name=tag+"_decs2sinput_"+str(fold_no))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = layers.Bidirectional(keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True,
                               name=tag+"_decs2sLstm1_"+str(fold_no)),
                                        name=tag+"_decBilstm1_"+str(fold_no))
    decoder_outputs, _,_,_,_ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_bnorm = keras.layers.BatchNormalization(name=tag+"_s2sBatch1_"+str(fold_no))
    decoder_outputs = decoder_bnorm(decoder_outputs)
    rshape=layers.Reshape([2*lstm_units],name=tag+"_s2sreshape1_"+str(fold_no))
    decoder_outputs = rshape(decoder_outputs) 
    
    decoder_dense = keras.layers.Dense(1, activation="sigmoid",
                                       name=tag+"_s2sDense1_"+str(fold_no))
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_grus2s_model_bnorm(input_shape, gru_units, loss, optimizer, metrics, tag, fold_no):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=input_shape,name=tag+"_encs2sinput_"+str(fold_no))
    encoder = layers.Bidirectional(layers.GRU(gru_units, return_state=True, name=tag+"_encs2sGru1_"+str(fold_no)),
                                   name=tag+"_encBigru1_"+str(fold_no))
    encoder_outputs, forw_state_h,back_state_h= encoder(encoder_inputs)
    
    # We discard `encoder_outputs` and only keep the states.
    final_enc_h = layers.Concatenate(name=tag+"_encConcat1_"+str(fold_no))([forw_state_h,back_state_h])
    
    # Set up the decoder, using `encoder_states` as initial state.
    
    decoder_inputs = keras.Input(shape=(1,1),name=tag+"_decs2sinput_"+str(fold_no))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_gru = keras.layers.GRU(2*gru_units, return_sequences=True, return_state=True,
                               name=tag+"_decs2sGru1_"+str(fold_no))
    decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=final_enc_h)
    decoder_bnorm = keras.layers.BatchNormalization(name=tag+"_s2sBatch1_"+str(fold_no))
    decoder_outputs = decoder_bnorm(decoder_outputs)
    rshape=layers.Reshape([2*gru_units],name=tag+"_s2sreshape1_"+str(fold_no))
    decoder_outputs = rshape(decoder_outputs) 
    
    decoder_dense = keras.layers.Dense(1, activation="sigmoid",
                                       name=tag+"_s2sDense1_"+str(fold_no))
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_grus2s_model_bnorm2(input_shape, gru_units, loss, optimizer, metrics, tag, fold_no):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=input_shape,name=tag+"_encs2sinput_"+str(fold_no))
    encoder = layers.Bidirectional(layers.GRU(gru_units, return_state=True, name=tag+"_encs2sGru1_"+str(fold_no)),
                                   name=tag+"_encBigru1_"+str(fold_no))
    encoder_outputs, forw_state_h,back_state_h= encoder(encoder_inputs)
    
    # We discard `encoder_outputs` and only keep the states.
    # final_enc_h = layers.Concatenate(name=tag+"_encConcat1_"+str(fold_no))([forw_state_h,back_state_h])
    encoder_states =[forw_state_h, back_state_h]
    # Set up the decoder, using `encoder_states` as initial state.
    
    decoder_inputs = keras.Input(shape=(1,1),name=tag+"_decs2sinput_"+str(fold_no))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_gru = layers.Bidirectional(keras.layers.GRU(gru_units, return_sequences=True, return_state=True,
                               name=tag+"_decs2sGru1_"+str(fold_no)),
                                       name=tag+"_decBigru1_"+str(fold_no))
    decoder_outputs, _,_ = decoder_gru(decoder_inputs, initial_state=encoder_states)
    decoder_bnorm = keras.layers.BatchNormalization(name=tag+"_s2sBatch1_"+str(fold_no))
    decoder_outputs = decoder_bnorm(decoder_outputs)
    rshape=layers.Reshape([2*gru_units],name=tag+"_s2sreshape1_"+str(fold_no))
    decoder_outputs = rshape(decoder_outputs) 
    
    decoder_dense = keras.layers.Dense(1, activation="sigmoid",
                                       name=tag+"_s2sDense1_"+str(fold_no))
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_lstms2sAttention_model_bnorm(input_shape, lstm_units, loss, optimizer, metrics, tag, fold_no):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=input_shape,name=tag+"_encinput_"+str(fold_no))
    
    # Bidirectional lstm layer
    enc_lstm1 = layers.Bidirectional(layers.LSTM(lstm_units,return_sequences=True,return_state=False,
                                                  name=tag+"_encLstm1_"+str(fold_no)),
                                      name=tag+"_encBilstm1_"+str(fold_no))
    encoder_outputs1 = enc_lstm1(encoder_inputs)
    encoder_bnorm = keras.layers.BatchNormalization(name=tag+"_encBatch1_"+str(fold_no))
    encoder_outputs1 = encoder_bnorm(encoder_outputs1)
    enc_lstm2 = layers.Bidirectional(layers.LSTM(lstm_units,return_sequences=True,return_state=False,
                                                  name=tag+"_encLstm2_"+str(fold_no)),
                                      name=tag+"_encBilstm2_"+str(fold_no))
    encoder_outputs2 = enc_lstm2(encoder_outputs1)
    encoder_bnorm = keras.layers.BatchNormalization(name=tag+"_encBatch2_"+str(fold_no))
    encoder_outputs2 = encoder_bnorm(encoder_outputs2)
    enc_lstm3 = layers.Bidirectional(layers.LSTM(lstm_units,return_sequences=True,return_state=True,
                                                  name=tag+"_Lstm3_"+str(fold_no)),
                                      name=tag+"_encBilstm3_"+str(fold_no))

    encoder_outputs3, forw_state_h, forw_state_c, back_state_h, back_state_c = enc_lstm3(encoder_inputs)
    encoder_bnorm = keras.layers.BatchNormalization(name=tag+"_encBatch1_"+str(fold_no))
    encoder_outputs3 = encoder_bnorm(encoder_outputs3)
    
    final_enc_h = layers.Concatenate(name=tag+"_encConcat1_"+str(fold_no))([forw_state_h,back_state_h])
    final_enc_c = layers.Concatenate(name=tag+"_encConcat2_"+str(fold_no))([forw_state_c,back_state_c])
    
    encoder_states =[final_enc_h, final_enc_c]
    
    # Set up the decoder, using `encoder_states` as initial state.

    decoder_inputs = keras.Input(shape=(1,1),name=tag+"_decinput_"+str(fold_no))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = layers.LSTM(2*lstm_units, return_sequences=True, return_state=True,
                               name=tag+"_decLstm1_"+str(fold_no))
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_bnorm = keras.layers.BatchNormalization(name=tag+"_decBatch1_"+str(fold_no))
    decoder_outputs = decoder_bnorm(decoder_outputs)
    #Attention Layer
    # attention_layer = tf.keras.layers.Attention()
    attention_layer = AttentionLayer(name=tag+"_atten_"+str(fold_no))
    attention_result,_= attention_layer([encoder_outputs3,decoder_outputs])
    # Concat attention output and decoder LSTM output 
    decoder_concat_input = keras.layers.Concatenate(axis=-1, name=tag+"_decConcat1_"+str(fold_no))([decoder_outputs, attention_result])
    rshape=layers.Reshape([4*lstm_units],name=tag+"_s2sreshape1_"+str(fold_no))
    decoder_concat_input = rshape(decoder_concat_input)
    decoder_dense = keras.layers.Dense(1, activation="sigmoid",
                                       name=tag+"_decDense1_"+str(fold_no))
    decoder_outputs = decoder_dense(decoder_concat_input)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_lstms2sAttention_model_bnorm2(input_shape, lstm_units, loss, optimizer, metrics, tag, fold_no):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=input_shape,name=tag+"_encinput_"+str(fold_no))
    
    # Bidirectional lstm layer
    enc_lstm3 = layers.LSTM(lstm_units,return_sequences=True,return_state=True,
                                                  name=tag+"_Lstm1_"+str(fold_no))
    # encoder_outputs3, forw_state_h, forw_state_c, back_state_h, back_state_c = enc_lstm3(encoder_inputs)
    encoder_outputs3, state_h, state_c = enc_lstm3(encoder_inputs)
    encoder_bnorm = keras.layers.BatchNormalization(name=tag+"_encBatch1_"+str(fold_no))
    encoder_outputs3 = encoder_bnorm(encoder_outputs3)

    encoder_states =[state_h, state_c]
    # Set up the decoder, using `encoder_states` as initial state.

    decoder_inputs = keras.Input(shape=(1,1),name=tag+"_decinput_"+str(fold_no))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True,
                               name=tag+"_decLstm1_"+str(fold_no))
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_bnorm = keras.layers.BatchNormalization(name=tag+"_decBatch1_"+str(fold_no))
    decoder_outputs = decoder_bnorm(decoder_outputs)
    #Attention Layer
    # attention_layer = tf.keras.layers.Attention()
    attention_layer = AttentionLayer(name=tag+"_atten_"+str(fold_no))
    attention_result,_= attention_layer([encoder_outputs3,decoder_outputs])
    # Concat attention output and decoder LSTM output 
    decoder_concat_input = keras.layers.Concatenate(axis=-1, name=tag+"_decConcat1_"+str(fold_no))([decoder_outputs, attention_result])
    rshape=layers.Reshape([2*lstm_units],name=tag+"_s2sreshape1_"+str(fold_no))
    decoder_concat_input = rshape(decoder_concat_input)
    decoder_dense = keras.layers.Dense(1, activation="sigmoid",
                                       name=tag+"_decDense1_"+str(fold_no))
    decoder_outputs = decoder_dense(decoder_concat_input)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_lstms2sAttention_model_bnorm3(input_shape, lstm_units, loss, optimizer, metrics, tag, fold_no):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=input_shape,name=tag+"_encinput_"+str(fold_no))
    
    # Bidirectional lstm layer
    enc_lstm = layers.Bidirectional(layers.LSTM(lstm_units,return_sequences=True,return_state=True,
                                                  name=tag+"_encLstm1_"+str(fold_no)),
                                      name=tag+"_encBilstm3_"+str(fold_no))
    encoder_outputs, forw_state_h, forw_state_c, back_state_h, back_state_c = enc_lstm(encoder_inputs)
    encoder_bnorm = keras.layers.BatchNormalization(name=tag+"_encBatch1_"+str(fold_no))
    encoder_outputs = encoder_bnorm(encoder_outputs)
    # Set up the decoder, using `encoder_states` as initial state.
    
    final_enc_h = layers.Concatenate(name=tag+"_encConcat1_"+str(fold_no))([forw_state_h,back_state_h])
    final_enc_c = layers.Concatenate(name=tag+"_encConcat2_"+str(fold_no))([forw_state_c,back_state_c])
    
    encoder_states =[final_enc_h, final_enc_c]
    
    decoder_inputs = keras.Input(shape=(1,1),name=tag+"_decinput_"+str(fold_no))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm= layers.LSTM(2*lstm_units, return_sequences=True, return_state=True,
                               name=tag+"_decLstm1_"+str(fold_no))
    decoder_outputs, _,_ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_bnorm = keras.layers.BatchNormalization(name=tag+"_decBatch1_"+str(fold_no))
    decoder_outputs = decoder_bnorm(decoder_outputs)
    #Attention Layer
    # attention_layer = tf.keras.layers.Attention()
    attention_layer = AttentionLayer(name=tag+"_atten_"+str(fold_no))
    attention_result,_= attention_layer([encoder_outputs,decoder_outputs])
    # Concat attention output and decoder LSTM output 
    decoder_concat_input = keras.layers.Concatenate(axis=-1, name=tag+"_decConcat1_"+str(fold_no))([decoder_outputs, attention_result])
    rshape=layers.Reshape([4*lstm_units],name=tag+"_s2sreshape1_"+str(fold_no))
    decoder_concat_input = rshape(decoder_concat_input)
    decoder_dense = keras.layers.Dense(1, activation="sigmoid",
                                       name=tag+"_decDense1_"+str(fold_no))
    decoder_outputs = decoder_dense(decoder_concat_input)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_grus2sAttention_model_bnorm(input_shape, gru_units, loss, optimizer, metrics, tag, fold_no):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=input_shape,name=tag+"_encinput_"+str(fold_no))
    
    # Bidirectional lstm layer
    enc_gru = layers.GRU(gru_units,return_sequences=True,return_state=True,
                                                  name=tag+"_encGru1_"+str(fold_no))
    # encoder_outputs3, forw_state_h, forw_state_c, back_state_h, back_state_c = enc_lstm3(encoder_inputs)
    encoder_outputs, state_h = enc_gru(encoder_inputs)
    encoder_bnorm = keras.layers.BatchNormalization(name=tag+"_encBatch1_"+str(fold_no))
    encoder_outputs = encoder_bnorm(encoder_outputs)

    # Set up the decoder, using `encoder_states` as initial state.

    decoder_inputs = keras.Input(shape=(1,1),name=tag+"_decinput_"+str(fold_no))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_gru= layers.GRU(gru_units, return_sequences=True, return_state=True,
                               name=tag+"_decGru1_"+str(fold_no))
    decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=state_h)
    decoder_bnorm = keras.layers.BatchNormalization(name=tag+"_decBatch1_"+str(fold_no))
    decoder_outputs = decoder_bnorm(decoder_outputs)
    #Attention Layer
    # attention_layer = tf.keras.layers.Attention()
    attention_layer = AttentionLayer(name=tag+"_atten_"+str(fold_no))
    attention_result,_= attention_layer([encoder_outputs,decoder_outputs])
    # Concat attention output and decoder LSTM output 
    decoder_concat_input = keras.layers.Concatenate(axis=-1, name=tag+"_decConcat1_"+str(fold_no))([decoder_outputs, attention_result])
    rshape=layers.Reshape([2*gru_units],name=tag+"_s2sreshape1_"+str(fold_no))
    decoder_concat_input = rshape(decoder_concat_input)
    decoder_dense = keras.layers.Dense(1, activation="sigmoid",
                                       name=tag+"_decDense1_"+str(fold_no))
    decoder_outputs = decoder_dense(decoder_concat_input)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_grus2sAttention_model_bnorm2(input_shape, gru_units, loss, optimizer, metrics, tag, fold_no):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=input_shape,name=tag+"_encinput_"+str(fold_no))
    
    # Bidirectional lstm layer
    enc_gru = layers.Bidirectional(layers.GRU(gru_units,return_sequences=True,return_state=True,
                                                  name=tag+"_encGru1_"+str(fold_no)),
                                                  name=tag+"_encBigru1_"+str(fold_no))
    # encoder_outputs3, forw_state_h, forw_state_c, back_state_h, back_state_c = enc_lstm3(encoder_inputs)
    encoder_outputs, forw_state_h, back_state_h = enc_gru(encoder_inputs)
    encoder_bnorm = keras.layers.BatchNormalization(name=tag+"_encBatch1_"+str(fold_no))
    encoder_outputs = encoder_bnorm(encoder_outputs)

    final_enc_h = layers.Concatenate(name=tag+"_encConcat1_"+str(fold_no))([forw_state_h,back_state_h])

    # Set up the decoder, using `encoder_states` as initial state.

    decoder_inputs = keras.Input(shape=(1,1),name=tag+"_decinput_"+str(fold_no))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_gru= layers.GRU(2*gru_units, return_sequences=True, return_state=True,
                               name=tag+"_decGru1_"+str(fold_no))
    decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=final_enc_h)
    decoder_bnorm = keras.layers.BatchNormalization(name=tag+"_decBatch1_"+str(fold_no))
    decoder_outputs = decoder_bnorm(decoder_outputs)
    #Attention Layer
    # attention_layer = tf.keras.layers.Attention()
    attention_layer = AttentionLayer(name=tag+"_atten_"+str(fold_no))
    attention_result,_= attention_layer([encoder_outputs,decoder_outputs])
    # Concat attention output and decoder LSTM output 
    decoder_concat_input = keras.layers.Concatenate(axis=-1, name=tag+"_decConcat1_"+str(fold_no))([decoder_outputs, attention_result])
    rshape=layers.Reshape([4*gru_units],name=tag+"_s2sreshape1_"+str(fold_no))
    decoder_concat_input = rshape(decoder_concat_input)
    decoder_dense = keras.layers.Dense(1, activation="sigmoid",
                                       name=tag+"_decDense1_"+str(fold_no))
    decoder_outputs = decoder_dense(decoder_concat_input)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_grus2sAttention_model_bnorm3(input_shape, gru_units, loss, optimizer, metrics, tag, fold_no):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=input_shape,name=tag+"_encinput_"+str(fold_no))
    
    # Bidirectional lstm layer
    enc_gru = layers.Bidirectional(layers.GRU(gru_units,return_sequences=True,return_state=True,
                                                  name=tag+"_encGru1_"+str(fold_no)),
                                                  name=tag+"_encBigru1_"+str(fold_no))
    # encoder_outputs3, forw_state_h, forw_state_c, back_state_h, back_state_c = enc_lstm3(encoder_inputs)
    encoder_outputs, forw_state_h, back_state_h = enc_gru(encoder_inputs)
    encoder_bnorm = keras.layers.BatchNormalization(name=tag+"_encBatch1_"+str(fold_no))
    encoder_outputs = encoder_bnorm(encoder_outputs)

    # final_enc_h = layers.Concatenate(name=tag+"_encConcat1_"+str(fold_no))([forw_state_h,back_state_h])

    # Set up the decoder, using `encoder_states` as initial state.
    encoder_states =[forw_state_h, back_state_h]
    
    decoder_inputs = keras.Input(shape=(1,1),name=tag+"_decinput_"+str(fold_no))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_gru= layers.Bidirectional(layers.GRU(gru_units, return_sequences=True, return_state=True,
                               name=tag+"_decGru1_"+str(fold_no)),
                                      name=tag+"_decBigru1_"+str(fold_no))
    decoder_outputs, _,_ = decoder_gru(decoder_inputs, initial_state=encoder_states)
    decoder_bnorm = keras.layers.BatchNormalization(name=tag+"_decBatch1_"+str(fold_no))
    decoder_outputs = decoder_bnorm(decoder_outputs)
    #Attention Layer
    # attention_layer = tf.keras.layers.Attention()
    attention_layer = AttentionLayer(name=tag+"_atten_"+str(fold_no))
    attention_result,_= attention_layer([encoder_outputs,decoder_outputs])
    # Concat attention output and decoder LSTM output 
    decoder_concat_input = keras.layers.Concatenate(axis=-1, name=tag+"_decConcat1_"+str(fold_no))([decoder_outputs, attention_result])
    rshape=layers.Reshape([4*gru_units],name=tag+"_s2sreshape1_"+str(fold_no))
    decoder_concat_input = rshape(decoder_concat_input)
    decoder_dense = keras.layers.Dense(1, activation="sigmoid",
                                       name=tag+"_decDense1_"+str(fold_no))
    decoder_outputs = decoder_dense(decoder_concat_input)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_lstms2s_model(input_shape, lstm_units, loss, optimizer, metrics, tag, fold_no):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=input_shape,name=tag+"_encs2sinput_"+str(fold_no))
    encoder = layers.LSTM(lstm_units, return_state=True, name=tag+"_encs2sLstm1_"+str(fold_no))
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    
    # Set up the decoder, using `encoder_states` as initial state.
    
    decoder_inputs = keras.Input(shape=(1,1),name=tag+"_decs2sinput_"+str(fold_no))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True,
                               name=tag+"_decs2sLstm1_"+str(fold_no))
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    rshape=layers.Reshape([lstm_units],name=tag+"_s2sreshape1_"+str(fold_no))
    decoder_outputs = rshape(decoder_outputs) 
    decoder_dense = keras.layers.Dense(1, activation="sigmoid",
                                       name=tag+"_s2sDense1_"+str(fold_no))
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_grus2s_model(input_shape, gru_units, loss, optimizer, metrics, tag, fold_no):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=input_shape,name=tag+"_encs2sinput_"+str(fold_no))
    encoder = layers.Bidirectional(layers.GRU(gru_units, return_state=True, name=tag+"_encs2sGru1_"+str(fold_no)),
                                   name=tag+"_encBigru1_"+str(fold_no))
    encoder_outputs, forw_state_h,back_state_h= encoder(encoder_inputs)
    
    # We discard `encoder_outputs` and only keep the states.
    final_enc_h = layers.Concatenate(name=tag+"_encConcat1_"+str(fold_no))([forw_state_h,back_state_h])
    
    # Set up the decoder, using `encoder_states` as initial state.
    
    decoder_inputs = keras.Input(shape=(1,1),name=tag+"_decs2sinput_"+str(fold_no))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_gru = keras.layers.GRU(2*gru_units, return_sequences=True, return_state=True,
                               name=tag+"_decs2sGru1_"+str(fold_no))
    decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=final_enc_h)
    rshape=layers.Reshape([2*gru_units],name=tag+"_s2sreshape1_"+str(fold_no))
    decoder_outputs = rshape(decoder_outputs) 
    
    decoder_dense = keras.layers.Dense(1, activation="sigmoid",
                                       name=tag+"_s2sDense1_"+str(fold_no))
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_lstms2sAttention_model(input_shape, lstm_units, loss, optimizer, metrics, tag, fold_no):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=input_shape,name=tag+"_encinput_"+str(fold_no))
    
    # Bidirectional lstm layer
    enc_lstm1 = layers.Bidirectional(layers.LSTM(lstm_units,return_sequences=True,return_state=False,
                                                 name=tag+"_encLstm1_"+str(fold_no)),
                                     name=tag+"_encBilstm1_"+str(fold_no))
    encoder_outputs1 = enc_lstm1(encoder_inputs)
    
    enc_lstm2 = layers.Bidirectional(layers.LSTM(lstm_units,return_sequences=True,return_state=False,
                                                 name=tag+"_encLstm2_"+str(fold_no)),
                                     name=tag+"_encBilstm2_"+str(fold_no))
    encoder_outputs2 = enc_lstm2(encoder_outputs1)
    
    enc_lstm3 = layers.Bidirectional(layers.LSTM(lstm_units,return_sequences=True,return_state=True,
                                                 name=tag+"_Lstm3_"+str(fold_no)),
                                     name=tag+"_encBilstm3_"+str(fold_no))
    encoder_outputs3, forw_state_h, forw_state_c, back_state_h, back_state_c = enc_lstm3(encoder_outputs2)
    
    final_enc_h = layers.Concatenate(name=tag+"_encConcat1_"+str(fold_no))([forw_state_h,back_state_h])
    final_enc_c = layers.Concatenate(name=tag+"_encConcat2_"+str(fold_no))([forw_state_c,back_state_c])
    
    encoder_states =[final_enc_h, final_enc_c]
    
    # Set up the decoder, using `encoder_states` as initial state.

    decoder_inputs = keras.Input(shape=(1,1),name=tag+"_decinput_"+str(fold_no))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = layers.LSTM(2*lstm_units, return_sequences=True, return_state=True,
                               name=tag+"_decLstm1_"+str(fold_no))
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    
    #Attention Layer
    # attention_layer = tf.keras.layers.Attention()
    attention_layer = AttentionLayer(name=tag+"_atten_"+str(fold_no))
    attention_result,_= attention_layer([encoder_outputs3,decoder_outputs])
    
    # Concat attention output and decoder LSTM output 
    decoder_concat_input = layers.Concatenate(axis=-1, name=tag+"_decConcat1_"+str(fold_no))([decoder_outputs, attention_result])
    rshape=layers.Reshape([4*lstm_units],name=tag+"_s2sreshape1_"+str(fold_no))
    decoder_concat_input = rshape(decoder_concat_input)
    decoder_dense = keras.layers.Dense(1, activation="sigmoid",
                                       name=tag+"_decDense1_"+str(fold_no))
    decoder_outputs = decoder_dense(decoder_concat_input)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_grus2sAttention_model(input_shape, gru_units, loss, optimizer, metrics, tag, fold_no):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=input_shape,name=tag+"_encinput_"+str(fold_no))
    
    # Bidirectional lstm layer
    enc_gru = layers.Bidirectional(layers.GRU(gru_units,return_sequences=True,return_state=True,
                                                  name=tag+"_encGru1_"+str(fold_no)),
                                                  name=tag+"_encBigru1_"+str(fold_no))
    # encoder_outputs3, forw_state_h, forw_state_c, back_state_h, back_state_c = enc_lstm3(encoder_inputs)
    encoder_outputs, forw_state_h, back_state_h = enc_gru(encoder_inputs)

    final_enc_h = layers.Concatenate(name=tag+"_encConcat1_"+str(fold_no))([forw_state_h,back_state_h])

    # Set up the decoder, using `encoder_states` as initial state.

    decoder_inputs = keras.Input(shape=(1,1),name=tag+"_decinput_"+str(fold_no))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_gru= layers.GRU(2*gru_units, return_sequences=True, return_state=True,
                               name=tag+"_decGru1_"+str(fold_no))
    decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=final_enc_h)
    #Attention Layer
    # attention_layer = tf.keras.layers.Attention()
    attention_layer = AttentionLayer(name=tag+"_atten_"+str(fold_no))
    attention_result,_= attention_layer([encoder_outputs,decoder_outputs])
    # Concat attention output and decoder LSTM output 
    decoder_concat_input = keras.layers.Concatenate(axis=-1, name=tag+"_decConcat1_"+str(fold_no))([decoder_outputs, attention_result])
    rshape=layers.Reshape([4*gru_units],name=tag+"_s2sreshape1_"+str(fold_no))
    decoder_concat_input = rshape(decoder_concat_input)
    decoder_dense = keras.layers.Dense(1, activation="sigmoid",
                                       name=tag+"_decDense1_"+str(fold_no))
    decoder_outputs = decoder_dense(decoder_concat_input)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def build_lstms2s_inference_model(model, tag, fold_no):
    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)
    
    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = keras.Input(shape=(encoder_outputs.shape[1],),name=tag+"_decs2sStateHinput1_"+str(fold_no))
    decoder_state_input_c = keras.Input(shape=(encoder_outputs.shape[1],),name=tag+"_decs2sStateCinput1_"+str(fold_no))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    # decoder_states = [state_h_dec, state_c_dec]
    rshape=model.layers[4]
    decoder_outputs = rshape(decoder_outputs) 
    decoder_dense = model.layers[5]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, decoder_outputs
    )
    return encoder_model, decoder_model

def build_lstms2sAttention_inference_model(model, tag, fold_no):
    encoder_inputs = model.input[0]  # input_1
    encoder_outputs3, _, _, _, _ = model.layers[3].output  # lstm_1
    final_enc_h = model.layers[4].output
    final_enc_c = model.layers[5].output
    encoder_states =[final_enc_h, final_enc_c]
    encoder_model = Model(encoder_inputs, outputs = [encoder_outputs3] + encoder_states)
    decoder_state_h = keras.Input(shape=(encoder_outputs3.shape[2],),name=tag+"_decStateHinput1_"+str(fold_no))
    decoder_state_c = keras.Input(shape=(encoder_outputs3.shape[2],),name=tag+"_decStateCinput2_"+str(fold_no))
    decoder_hidden_state_input = keras.Input(shape=encoder_outputs3.shape[1:],
                                             name=tag+"_decHidStateinput1_"+str(fold_no))
    
    dec_states = [decoder_state_h, decoder_state_c]
    decoder_inputs = model.input[1]  # input_2 
    decoder_lstm = model.layers[7]
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(decoder_inputs, initial_state=dec_states)
    
    # Attention inference
    attention_layer = model.layers[8]
    attention_result_inf,_ = attention_layer([decoder_hidden_state_input, decoder_outputs2])
    
    decoder_concat_input_inf = layers.Concatenate(axis=-1, name=tag+"_decConcatinf1_"+str(fold_no))([decoder_outputs2, attention_result_inf])
    
    # dec_states2= [state_h2, state_c2]
    rshape=model.layers[10]
    decoder_concat_input_inf = rshape(decoder_concat_input_inf)
    decoder_dense = model.layers[11]
    decoder_outputs3 = decoder_dense(decoder_concat_input_inf)
    
    decoder_model= Model([decoder_inputs] + [decoder_hidden_state_input, decoder_state_h, decoder_state_c],
                         decoder_outputs3)
    return encoder_model, decoder_model

def build_gru_model(input_shape,gru_units,dense_units,dropout_rate,
                    optimizer,loss,metrics,tag,fold_no):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape, 
                                name=tag+"_input_"+str(fold_no)))
    model.add(layers.Bidirectional(layers.GRU(gru_units,return_sequences=True,
                                              name=tag+"_gru1_"+str(fold_no)),
                                   name=tag+"_bigru1_"+str(fold_no)))
    model.add(layers.Bidirectional(layers.GRU(gru_units,return_sequences=True,
                                              name=tag+"_gru2_"+str(fold_no)),
                                   name=tag+"_bigru2_"+str(fold_no)))
    model.add(layers.Bidirectional(layers.GRU(gru_units,return_sequences=True,
                                              name=tag+"_gru3_"+str(fold_no)),
                                   name=tag+"_bigru3_"+str(fold_no)))
    model.add(layers.Bidirectional(layers.GRU(gru_units,return_sequences=False,
                                              name=tag+"_gru4_"+str(fold_no)),
                                   name=tag+"_bigru4_"+str(fold_no)))
    # model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu',
              name=tag+"_dense1_"+str(fold_no)))
    model.add(layers.Dropout(dropout_rate,
              name=tag+"_dropout_"+str(fold_no)))
    model.add(layers.Dense(1,activation='sigmoid',
              name=tag+"_dense2_"+str(fold_no)))
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model

def build_gru_model2(input_shape,gru_units,dense_units,dropout_rate,
                    optimizer,loss,metrics,tag,fold_no):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape, 
                                name=tag+"_input_"+str(fold_no)))
    model.add(layers.Bidirectional(layers.GRU(gru_units,return_sequences=True,
                                              name=tag+"_gru1_"+str(fold_no)),
                                   name=tag+"_bigru1_"+str(fold_no)))
    model.add(layers.Flatten(name=tag+"_flatten_"+str(fold_no)))
    model.add(layers.Dense(dense_units, activation='relu',
              name=tag+"_dense1_"+str(fold_no)))
    model.add(layers.Dropout(dropout_rate,
              name=tag+"_dropout1_"+str(fold_no)))
    model.add(layers.Dense(1,activation='sigmoid',
              name=tag+"_dense2_"+str(fold_no)))
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model

def build_cnn_model(input_shape,n_filter,kernel_size,strides,padding,
                    dense_units,max_pool,dropout_rate,optimizer,loss,metrics,tag,fold_no):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape, 
                                name=tag+"_input_"+str(fold_no)))
    model.add(layers.Reshape((input_shape[0],input_shape[1],1), 
                             input_shape=input_shape,name=tag+"_reshape_"+str(fold_no)))
    model.add(layers.Conv2D(n_filter, kernel_size=kernel_size, 
                            strides=strides, padding=padding, 
                            activation='relu', 
                            name=tag+"_conv2d1_"+str(fold_no)))
    model.add(layers.MaxPooling2D(max_pool,padding=padding, 
                                  name=tag+"_maxpool2d1_"+str(fold_no)))
    model.add(layers.Flatten(name=tag+"_flatten_"+str(fold_no)))
    model.add(layers.Dense(dense_units, activation='relu',
                           name=tag+"_dense1_"+str(fold_no)))
    model.add(layers.Dropout(dropout_rate,
              name=tag+"_dropout1_"+str(fold_no)))
    model.add(layers.Dense(1,activation='sigmoid',
              name=tag+"_dense2_"+str(fold_no)))
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model

def build_ensemble_model_rc(results_dir,n_models_rnn,n_models_cnn,dense_units,
                         dropout_rate,optimizer,loss,metrics):
    models_rnn=load_models2(results_dir+'/model_rnn_fold',n_models_rnn)
    models_cnn=load_models2(results_dir+'/model_cnn_fold',n_models_cnn)
    
    ensemble_inputs, ensemble_outputs = create_model(models_rnn,"rnn")
    ensemble_inputs2, ensemble_outputs2 = create_model(models_cnn,"cnn")
    ensemble_inputs.extend(ensemble_inputs2)
    ensemble_outputs.extend(ensemble_outputs2)

    merged_output = concatenate(ensemble_outputs)   
    
    layer = layers.Dense(dense_units, activation='relu')(merged_output)
    layer = layers.Dropout(dropout_rate)(layer)
    output = layers.Dense(1,activation='sigmoid')(layer)
    model = Model(ensemble_inputs, output)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model

def build_ensemble_model_rcsa(results_dir,n_models_rnn,n_models_cnn,
                          n_models_ls2satt,n_models_ls2s,dense_units,
                         dropout_rate,optimizer,loss,metrics,fold_no):
    models_rnn=load_models2(results_dir+'/model_rnn_fold',n_models_rnn)
    models_cnn=load_models2(results_dir+'/model_cnn_fold',n_models_cnn)
    models_ls2s = load_models2(results_dir+'/model_lstms2s_fold',n_models_ls2s)
    models_ls2satt = load_models2(results_dir+'/model_lstms2satt_fold',n_models_ls2satt)

    ensemble_inputs, ensemble_outputs = create_model(models_rnn,"rnn")
    ensemble_inputs2, ensemble_outputs2 = create_model(models_cnn,"cnn")
    ensemble_inputs3, ensemble_outputs3 = create_model2(models_ls2s,"lstms2s")
    ensemble_inputs4, ensemble_outputs4 = create_model2(models_ls2satt,"lstms2satt")
    ensemble_inputs.extend(ensemble_inputs2)
    ensemble_inputs.extend(ensemble_inputs3)
    ensemble_inputs.extend(ensemble_inputs4)
    ensemble_outputs.extend(ensemble_outputs2)
    ensemble_outputs.extend(ensemble_outputs3)
    ensemble_outputs.extend(ensemble_outputs4)

    merged_output = concatenate(ensemble_outputs)   
    
    layer = layers.Dense(dense_units, activation='relu')(merged_output)
    layer = layers.Dropout(dropout_rate)(layer)
    output = layers.Dense(1,activation='sigmoid')(layer)
    model = Model(ensemble_inputs, output)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model

def build_ensemble_model_s(results_dir,n_models_rnn,n_models_cnn,
                          n_models_ls2satt,n_models_ls2s,dense_units,
                         dropout_rate,optimizer,loss,metrics,fold_no):
    # models_rnn=load_models2(results_dir+'/model_rnn_fold',n_models_rnn)
    # models_cnn=load_models2(results_dir+'/model_cnn_fold',n_models_cnn)
    models_ls2s = load_models2(results_dir+'/model_lstms2s_fold',n_models_ls2s)
    # models_ls2satt = load_models2(results_dir+'/model_lstms2satt_fold',n_models_ls2satt)

    # ensemble_inputs, ensemble_outputs = create_model(models_rnn,"rnn")
    # ensemble_inputs2, ensemble_outputs2 = create_model(models_cnn,"cnn")
    ensemble_inputs3, ensemble_outputs3 = create_model2(models_ls2s,"lstms2s")
    # ensemble_inputs4, ensemble_outputs4 = create_model2(models_ls2satt,"lstms2satt")
    # ensemble_inputs.extend(ensemble_inputs2)
    # ensemble_inputs.extend(ensemble_inputs3)
    # ensemble_inputs3.extend(ensemble_inputs4)
    # ensemble_outputs.extend(ensemble_outputs2)
    # ensemble_outputs.extend(ensemble_outputs3)
    # ensemble_outputs3.extend(ensemble_outputs4)

    merged_output = concatenate(ensemble_outputs3)   
    
    layer = layers.Dense(dense_units, activation='relu')(merged_output)
    bnorm = keras.layers.BatchNormalization(name="ensbBatch1_"+str(fold_no))
    layer = bnorm(layer)
    layer = layers.Dropout(dropout_rate)(layer)
    output = layers.Dense(1,activation='sigmoid')(layer)
    
    model = Model(ensemble_inputs3, output)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model

def build_ensemble_model_rc2(results_dir,input_shape,dense_units,
                         dropout_rate,optimizer,loss,metrics,tag,fold_no):
    
    model_folder_rnn=results_dir+f'/model_rnn_fold/fold_{fold_no}/model.tf'
    model_rnn = keras.models.load_model(model_folder_rnn)
    model_folder_cnn=results_dir+f'/model_cnn_fold/fold_{fold_no}/model.tf'
    model_cnn = keras.models.load_model(model_folder_cnn)
    for layer in model_rnn.layers:
        # make layers not trainable
        layer.trainable = False
    for layer in model_cnn.layers:
        # make layers not trainable
        layer.trainable = False
    # print(model_rnn.layers)
    layer_input=keras.Input(shape=input_shape, 
                                name=tag+"_input_"+str(fold_no))

    layer_model_cnn=model_cnn.layers[0](layer_input)
    layer_model_cnn=model_cnn.layers[1](layer_model_cnn)
    layer_model_cnn=model_cnn.layers[2](layer_model_cnn)
    layer_model_cnn=layers.Flatten(name=tag+"_flatten1_"+str(fold_no))(layer_model_cnn)
    layer_model_cnn=layers.Dropout(dropout_rate,name=tag+"_dropout1_"+str(fold_no))(layer_model_cnn)
    
    layer_model_rnn=model_rnn.layers[0](layer_input)
    layer_model_rnn=layers.Dropout(dropout_rate,name=tag+"_dropout2_"+str(fold_no))(layer_model_rnn)
    layer_model_rnn=layers.Flatten(name=tag+"_flatten2_"+str(fold_no))(layer_model_rnn)
    
    merged_output = concatenate([layer_model_rnn, layer_model_cnn],name=tag+"_concat1_"+str(fold_no))   
    model_ensemble = layers.Dropout(dropout_rate,name=tag+"_dropout3_"+str(fold_no))(merged_output)
    model_ensemble = layers.Dense(dense_units, activation='relu',
                                  name=tag+"_dense1_"+str(fold_no))(model_ensemble)
    model_ensemble = layers.Dropout(dropout_rate,name=tag+"_dropout4_"+str(fold_no))(model_ensemble)
    model_ensemble = layers.Dense(1, activation='sigmoid',
                                  name=tag+"_dense2_"+str(fold_no))(model_ensemble)
    
    model = Model(layer_input.output, model_ensemble)
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model

def build_ensemble_model_rc3(results_dir,input_shape,dense_units,
                         dropout_rate,optimizer,loss,metrics,tag,fold_no):
    
    model_folder_rnn=results_dir+f'/model_rnn_fold/fold_{fold_no}/model.tf'
    model_rnn = keras.models.load_model(model_folder_rnn)
    model_folder_cnn=results_dir+f'/model_cnn_fold/fold_{fold_no}/model.tf'
    model_cnn = keras.models.load_model(model_folder_cnn)
    for layer in model_rnn.layers:
        # make layers not trainable
        layer.trainable = False
    for layer in model_cnn.layers:
        # make layers not trainable
        layer.trainable = False
    # print(model_rnn.layers)

    layer_model_rnn=model_rnn.layers[0](model_rnn.input)
    layer_model_rnn=model_rnn.layers[1](layer_model_rnn)
    layer_model_rnn=model_rnn.layers[2](layer_model_rnn)

    layer_model_cnn=model_cnn.layers[0](model_cnn.input)
    layer_model_cnn=model_cnn.layers[1](layer_model_cnn)
    layer_model_cnn=model_cnn.layers[2](layer_model_cnn)
    layer_model_cnn=model_cnn.layers[3](layer_model_cnn)
    layer_model_cnn=model_cnn.layers[4](layer_model_cnn)
    
    merged_output = concatenate([layer_model_rnn, layer_model_cnn])   
    model_ensemble = layers.Dropout(dropout_rate)(merged_output)
    model_ensemble = layers.Dense(dense_units, activation='relu')(model_ensemble)
    model_ensemble = layers.Dropout(dropout_rate)(model_ensemble)
    model_ensemble = layers.Dense(1, activation='sigmoid')(model_ensemble)
    
    model = Model([model_rnn.input,model_cnn.input], model_ensemble)
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model

def build_ensemble_model(results_dir,models_tags_num,dense_units,
                         dropout_rate,optimizer,loss,metrics,fold_no):
    ensemble_inputs=[]
    ensemble_outputs=[]
    for tag,n in models_tags_num.items():
        if n!=0:
            models=load_models2(results_dir+'/model_'+tag+'_fold',n)
            inputs, outputs = create_model2(models,tag)
            ensemble_inputs.extend(inputs)
            ensemble_outputs.extend(outputs)
            
    merged_output = concatenate(ensemble_outputs)   
    
    layer = layers.Dense(dense_units, activation='relu')(merged_output)
    bnorm = keras.layers.BatchNormalization(name="ensbBatch1_"+str(fold_no))
    layer = bnorm(layer)
    layer = layers.Dropout(dropout_rate)(layer)
    output = layers.Dense(1,activation='sigmoid')(layer)
    
    model = Model(ensemble_inputs, output)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model

def build_ensemble_model2(results_dir,models_tags_num,input_shape,dense_units,
                         dropout_rate,optimizer,loss,metrics,tag,fold_no):
    layer_input_1=keras.Input(shape=input_shape, 
                                name="input1")
    layer_input_2=keras.Input(shape=(1,1), 
                                name="input2")
    ensemble_outputs=[]
    for tag_mdl,n in models_tags_num.items():
        if n!=0:
            models=load_models2(results_dir+'/model_'+tag_mdl+'_fold',n)
            outputs = create_model3(models,layer_input_1,layer_input_2,tag_mdl)
            ensemble_outputs.extend(outputs)
            
    merged_output = concatenate(ensemble_outputs)   
    
    layer = layers.Dense(dense_units, activation='relu',
                         name=tag+"dense1_"+str(fold_no))(merged_output)
    bnorm = keras.layers.BatchNormalization(name=tag+"Batch1_"+str(fold_no))(layer)
    layer = layers.Dropout(dropout_rate,name=tag+"dropout1_"+str(fold_no))(bnorm)
    output = layers.Dense(1,activation='sigmoid',
                          name=tag+"dense2_"+str(fold_no))(layer)
    
    model = Model([layer_input_1,layer_input_2], output)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model

def build_ensemble_model3(results_dir,models_tags_num,input_shape,dense_units,
                         dropout_rate,optimizer,loss,metrics,tag,fold_no):
    layer_input_1=keras.Input(shape=input_shape, 
                                name="input1")
    layer_input_2=keras.Input(shape=(1,1), 
                                name="input2")
    ensemble_outputs=[]
    for tag_mdl,n in models_tags_num.items():
        if n!=0:
            models=load_models2(results_dir+'/model_'+tag_mdl+'_fold',n)
            outputs = create_model4(models,layer_input_1,layer_input_2,tag_mdl)
            ensemble_outputs.extend(outputs)
            
    merged_output = concatenate(ensemble_outputs)   
    
    layer = layers.Dense(4*dense_units, activation='relu',
                         name=tag+"dense1_"+str(fold_no))(merged_output)
    bnorm = keras.layers.BatchNormalization(name=tag+"Batch1_"+str(fold_no))(layer)
    layer = layers.Dropout(dropout_rate,name=tag+"dropout1_"+str(fold_no))(bnorm)
    layer = layers.Dense(dense_units, activation='relu',
                          name=tag+"dense2_"+str(fold_no))(layer)
    bnorm = keras.layers.BatchNormalization(name=tag+"Batch2_"+str(fold_no))(layer)
    layer = layers.Dropout(dropout_rate,name=tag+"dropout2_"+str(fold_no))(bnorm)
    output = layers.Dense(1,activation='sigmoid',
                          name=tag+"dense3_"+str(fold_no))(layer)
    
    model = Model([layer_input_1,layer_input_2], output)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model

def load_models(path,n_models):
    models=[]
    for i in range(1,n_models+1):
        models.append(keras.models.load_model(path+f'/fold_{i}/fold_{i}.tf'))
    return models

def load_models2(path,n_models):
    models=[]
    for i in range(1,n_models+1):
        models.append(keras.models.load_model(path+f'/fold_{i}/model.tf'))
    return models

def create_model(models,tag):
    for i in range(len(models)):
        model = models[i]
        # model.input._name=tag + '_' + str(i+1) + '_' + model.input.name
        # rename(model, model.inputs[0], tag + '_' + str(i+1))
        for layer in model.layers:
            # make layers not trainable
            layer.trainable = False
            # rename layers
            # layer._name =tag + '_ensemble_' + str(i+1) + '_' + layer.name #str(random.randrange(1,101)) + '_' 
            # rename(model, layer, tag + '_ensemble_' + str(i+1) + '_' + layer.name)
            # print(layer._name)
        # print(model.layers)
    # multi-headed input
    # print([model.input._name for model in models])
    # print([model.inputs[:] for model in models])
    ensemble_inputs = [model.input for model in models]
    # output list of models
    ensemble_outputs = [model.output for model in models]
    return ensemble_inputs, ensemble_outputs

def create_model2(models,tag):
    ensemble_inputs=[]
    ensemble_outputs=[]
    for i in range(len(models)):
        model = models[i]
        # model.input._name=tag + '_' + str(i+1) + '_' + model.input.name
        # rename(model, model.inputs[0], tag + '_' + str(i+1))
        for layer in model.layers:
            # make layers not trainable
            layer.trainable = False
            # rename layers
            # layer._name =tag + '_ensemble_' + str(i+1) + '_' + layer.name #str(random.randrange(1,101)) + '_' 
            # rename(model, layer, tag + '_ensemble_' + str(i+1) + '_' + layer.name)
            # print(layer._name)
        if 's2s' in tag: 
            ensemble_inputs.append(model.input[:2])
        else:
            ensemble_inputs.append(model.input)
        ensemble_outputs.append(model.output)
        # print(model.layers)
    # multi-headed input
    # print([model.input._name for model in models])
    # print([model.inputs[:] for model in models])
    # ensemble_inputs = [model.input for model in models]
    # output list of models
    
    # ensemble_outputs = [model.output for model in models]
    return ensemble_inputs, ensemble_outputs

def create_model3(models,layer_input_1,layer_input_2,tag):
    ensemble_outputs=[]
    for i in range(len(models)):
        model = models[i]
        for layer in model.layers:
            # make layers not trainable
            layer.trainable = False
        if 's2s' in tag: 
            if 'att' in tag: 
                model_output=outputs_lstms2satt_bnorm(layer_input_1,layer_input_2,model)
            else:
                model_output=outputs_lstms2s(layer_input_1,layer_input_2,model)
        else:
            model_output=outputs_dl(layer_input_1,model)
        ensemble_outputs.append(model_output)
    return  ensemble_outputs

def create_model4(models,layer_input_1,layer_input_2,tag):
    ensemble_outputs=[]
    for i in range(len(models)):
        model = models[i]
        for layer in model.layers:
            # make layers not trainable
            layer.trainable = False
        if 's2s' in tag: 
            if 'att' in tag: 
                if 'gru' in tag: model_output=outputs_grus2satt_bnorm2(layer_input_1,layer_input_2,model)
                if 'lstm' in tag: model_output=outputs_lstms2satt_bnorm2(layer_input_1,layer_input_2,model)
            else:
                if 'gru' in tag: model_output=outputs_grus2s2(layer_input_1,layer_input_2,model)
                if 'lstm' in tag: model_output=outputs_lstms2s2(layer_input_1,layer_input_2,model)
        else:
            model_output=outputs_dl2(layer_input_1,model)
        ensemble_outputs.append(model_output)
    return  ensemble_outputs

def build_ensemble_model_lstms2s(results_dir,models,tag,input_shape,dense_units,
                  fold_no,dropout_rate,optimizer,loss,metrics):
    ensemble_outputs=[]
    tag='lstms2s'
    models=load_models2(results_dir+'/model_'+tag+'_fold',5)
    layer_input_1=keras.Input(shape=input_shape, 
                                name="input1")
    layer_input_2=keras.Input(shape=(1,1), 
                                name="input2")
    for i in range(len(models)):
        model = models[i]
        print(model.layers)
        # model.input._name=tag + '_' + str(i+1) + '_' + model.input.name
        # rename(model, model.inputs[0], tag + '_' + str(i+1))
        for layer in model.layers:
            # make layers not trainable
            layer.trainable = False
            # rename layers
            # layer._name =tag + '_ensemble_' + str(i+1) + '_' + layer.name #str(random.randrange(1,101)) + '_' 
            # rename(model, layer, tag + '_ensemble_' + str(i+1) + '_' + layer.name)
            # print(layer._name)
        model_output=outputs_lstms2s(layer_input_1,layer_input_2,model)
        ensemble_outputs.append(model_output)
        # print(model.layers)
    # multi-headed input
    # print([model.input._name for model in models])
    # print([model.inputs[:] for model in models])
    # ensemble_inputs = [model.input for model in models]
    # output list of models
    merged_output = concatenate(ensemble_outputs)   
    layer = layers.Dense(dense_units, activation='relu')(merged_output)
    bnorm = keras.layers.BatchNormalization(name="ensbBatch1_"+str(fold_no))
    layer = bnorm(layer)
    layer = layers.Dropout(dropout_rate)(layer)
    output = layers.Dense(1,activation='sigmoid')(layer)
    
    ensemble_model = Model([layer_input_1,layer_input_2], output)
    ensemble_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return ensemble_model

def build_ensemble_model_lstms2satt_bnorm(results_dir,models,tag,input_shape,dense_units,
                  fold_no,dropout_rate,optimizer,loss,metrics):
    ensemble_outputs=[]
    tag='lstms2satt'
    models=load_models2(results_dir+'/model_'+tag+'_fold',5)
    layer_input_1=keras.Input(shape=input_shape, 
                                name="input1")
    layer_input_2=keras.Input(shape=(1,1), 
                                name="input2")
    for i in range(len(models)):
        model = models[i]
        # print(model.layers)
        # model.input._name=tag + '_' + str(i+1) + '_' + model.input.name
        # rename(model, model.inputs[0], tag + '_' + str(i+1))
        for layer in model.layers:
            # make layers not trainable
            layer.trainable = False
            # rename layers
            # layer._name =tag + '_ensemble_' + str(i+1) + '_' + layer.name #str(random.randrange(1,101)) + '_' 
            # rename(model, layer, tag + '_ensemble_' + str(i+1) + '_' + layer.name)
            # print(layer._name)
        # print(model.layers)
        model_output=outputs_lstms2satt_bnorm(layer_input_1,layer_input_2,model)
        ensemble_outputs.append(model_output)
        # print(model.layers)
    # multi-headed input
    # print([model.input._name for model in models])
    # print([model.inputs[:] for model in models])
    # ensemble_inputs = [model.input for model in models]
    # output list of models
    # print(ensemble_outputs)
    merged_output = concatenate(ensemble_outputs)   
    layer = layers.Dense(dense_units, activation='relu')(merged_output)
    bnorm = keras.layers.BatchNormalization(name="ensbBatch1_"+str(fold_no))
    layer = bnorm(layer)
    layer = layers.Dropout(dropout_rate)(layer)
    output = layers.Dense(1,activation='sigmoid')(layer)
    
    model = Model([layer_input_1,layer_input_2], output)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model

def outputs_lstms2satt_bnorm(layer_input_1,layer_input_2,model):
    encoder_outputs = model.layers[1](layer_input_1)
    encoder_outputs = model.layers[2](encoder_outputs)
    encoder_outputs = model.layers[3](encoder_outputs)
    encoder_outputs = model.layers[4](encoder_outputs)
    encoder_outputs, forw_state_h, forw_state_c, back_state_h, back_state_c = model.layers[5](encoder_outputs) # lstm_1

    final_enc_h = model.layers[7]([forw_state_h,back_state_h])
    final_enc_c = model.layers[8]([forw_state_c,back_state_c])
    # final_enc_h = model.layers[7].output
    # final_enc_c = model.layers[8].output
    encoder_states =[final_enc_h, final_enc_c]
    
    decoder_lstm = model.layers[9]
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(layer_input_2, initial_state=encoder_states)
    encoder_outputs=model.layers[11](encoder_outputs)
    decoder_outputs2=model.layers[10](decoder_outputs2)
    
    # Attention inference
    attention_layer = model.layers[12]
    attention_result_inf,_ = attention_layer([encoder_outputs, decoder_outputs2])
    
    # decoder_concat_input_inf = layers.Concatenate(axis=-1, name=tag+"_decConcatinf1_"+str(i))([decoder_outputs2, attention_result_inf])
    decoder_concat_input_inf = model.layers[13]([decoder_outputs2, attention_result_inf])
    # dec_states2= [state_h2, state_c2]
    # rshape=model.layers[10]
    # decoder_concat_input_inf = rshape(decoder_concat_input_inf)
    decoder_outputs = model.layers[14](decoder_concat_input_inf)
    decoder_outputs = model.layers[15](decoder_outputs)
    return decoder_outputs

def outputs_lstms2satt_bnorm2(layer_input_1,layer_input_2,model):
    encoder_outputs = model.layers[1](layer_input_1)
    encoder_outputs = model.layers[2](encoder_outputs)
    encoder_outputs = model.layers[3](encoder_outputs)
    encoder_outputs = model.layers[4](encoder_outputs)
    encoder_outputs, forw_state_h, forw_state_c, back_state_h, back_state_c = model.layers[5](encoder_outputs) # lstm_1

    final_enc_h = model.layers[7]([forw_state_h,back_state_h])
    final_enc_c = model.layers[8]([forw_state_c,back_state_c])
    # final_enc_h = model.layers[7].output
    # final_enc_c = model.layers[8].output
    encoder_states =[final_enc_h, final_enc_c]
    
    decoder_lstm = model.layers[9]
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(layer_input_2, initial_state=encoder_states)
    encoder_outputs=model.layers[11](encoder_outputs)
    decoder_outputs2=model.layers[10](decoder_outputs2)
    
    # Attention inference
    attention_layer = model.layers[12]
    attention_result_inf,_ = attention_layer([encoder_outputs, decoder_outputs2])
    
    # decoder_concat_input_inf = layers.Concatenate(axis=-1, name=tag+"_decConcatinf1_"+str(i))([decoder_outputs2, attention_result_inf])
    decoder_concat_input_inf = model.layers[13]([decoder_outputs2, attention_result_inf])
    # dec_states2= [state_h2, state_c2]
    # rshape=model.layers[10]
    # decoder_concat_input_inf = rshape(decoder_concat_input_inf)
    decoder_outputs = model.layers[14](decoder_concat_input_inf)
    return decoder_outputs

def outputs_grus2satt_bnorm2(layer_input_1,layer_input_2,model):
    encoder_outputs, forw_state_h, back_state_h = model.layers[1](layer_input_1) # lstm_1

    final_enc_h = model.layers[3]([forw_state_h,back_state_h])

    decoder_gru = model.layers[4]
    decoder_outputs2, _ = decoder_gru(layer_input_2, initial_state=final_enc_h)
    encoder_outputs=model.layers[6](encoder_outputs)
    decoder_outputs2=model.layers[5](decoder_outputs2)
    
    # Attention layer
    attention_layer = model.layers[7]
    attention_result,_ = attention_layer([encoder_outputs, decoder_outputs2])
    
    # decoder_concat_input_inf = layers.Concatenate(axis=-1, name=tag+"_decConcatinf1_"+str(i))([decoder_outputs2, attention_result_inf])
    decoder_concat_input = model.layers[8]([decoder_outputs2, attention_result])
    # dec_states2= [state_h2, state_c2]
    # rshape=model.layers[10]
    # decoder_concat_input_inf = rshape(decoder_concat_input_inf)
    decoder_outputs = model.layers[9](decoder_concat_input)
    return decoder_outputs

def outputs_lstms2s(layer_input_1,layer_input_2,model):
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2](layer_input_1)  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        layer_input_2, initial_state=encoder_states
    )
    for i in range(4,len(model.layers)):
        decoder_outputs = model.layers[i](decoder_outputs)
    return decoder_outputs

def outputs_lstms2s2(layer_input_1,layer_input_2,model):
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2](layer_input_1)  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        layer_input_2, initial_state=encoder_states
    )
    for i in range(4,len(model.layers)-1):
        decoder_outputs = model.layers[i](decoder_outputs)
    return decoder_outputs

def outputs_grus2s2(layer_input_1,layer_input_2,model):
    encoder_outputs, forw_state_h_enc, back_state_h_enc = model.layers[1](layer_input_1)  # lstm_1
    final_enc_h = model.layers[3]([forw_state_h_enc, back_state_h_enc])

    decoder_gru = model.layers[4]
    decoder_outputs, state_h_dec = decoder_gru(
        layer_input_2, initial_state=final_enc_h
    )
    for i in range(5,len(model.layers)-1):
        decoder_outputs = model.layers[i](decoder_outputs)
    return decoder_outputs

def outputs_dl(layer_input_1,model):
    layer=model.layers[0](layer_input_1)
    for i in range(1,len(model.layers)):
        layer=model.layers[i](layer)
    return layer

def outputs_dl2(layer_input_1,model):
    layer=model.layers[0](layer_input_1)
    for i in range(1,len(model.layers)-1):
        layer=model.layers[i](layer)
    return layer

def rename(model, layer, new_name):
    def _get_node_suffix(name):
        for old_name in old_nodes:
            if old_name.startswith(name):
                return old_name[len(name):]

    old_name = layer.name
    old_nodes = list(model._network_nodes)
    new_nodes = []

    for l in model.layers:
        if l.name == old_name:
            l._name = new_name
            # vars(l).__setitem__('_name', new)  # bypasses .__setattr__
            new_nodes.append(new_name + _get_node_suffix(old_name))
        else:
            new_nodes.append(l.name + _get_node_suffix(l.name))
    model._network_nodes = set(new_nodes)

def classWeight_comp(class_num,y,sqrtw,classwcomp):
    cl_w=class_weight.compute_class_weight(class_weight='balanced',classes=class_num,y=y)
    w0=cl_w[0]
    w1=cl_w[1]
    # w0=train.shape[0]/(2*(train.shape[0]-np.sum(target_train,dtype=int)))
    # print(train.shape[0],np.sum(target_train,dtype=int))
    # w1=train.shape[0]/(2*np.sum(target_train,dtype=int))
    classWeight={0: w0, 1: w1}
    #classWeight={0: 0.55, 1: 4.97}
    if sqrtw==True:
    	w0=math.sqrt(w0)
    	w1=math.sqrt(w1)
    if classwcomp==True: 
        classWeight={0: w0, 1: w1}
    else:
        classWeight={0: 0.55, 1: 4.97}
    return classWeight

def sampleWeight_comp(classWeight, y, sampw):
    if sampw==True:
        return np.array([classWeight[0] if i==0 else classWeight[1] 
                     for i in y.tolist()])
    else: 
        return None
    
def print_validation_results_fold(model,scores,fold_no):
    print(f'Validation Scores for fold {fold_no}: {model.metrics_names[0]} : {scores[0]}; {model.metrics_names[1]} : {scores[1]}')
    print(f'                          : {model.metrics_names[2]} : {scores[2]}; {model.metrics_names[3]} : {scores[3]}')
    print(f'                          : {model.metrics_names[4]} : {scores[4]}; {model.metrics_names[5]} : {scores[5]}')
    print(f'                          : {model.metrics_names[6]} : {scores[6]}')
    print(f'                          : {model.metrics_names[7]} : {int(scores[7])}; {model.metrics_names[8]} : {int(scores[8])}')
    print(f'                          : {model.metrics_names[9]} : {int(scores[9])}; {model.metrics_names[10]} : {int(scores[10])}')
    
def compute_metrics(model,scores):
    TP=scores[model.metrics_names.index('true_positives')]
    TN=scores[model.metrics_names.index('true_negatives')]
    FP=scores[model.metrics_names.index('false_positives')]
    FN=scores[model.metrics_names.index('false_negatives')]
    try:
        specivity_sc=TN/(TN+FP)
    except:
        specivity_sc=0
    try:
        recall_sc=TP/(TP+FN)
    except:
        recall_sc=0
    try:
        precision_sc=TP/(TP+FP)
    except:
        precision_sc=0
    try:
        f1_sc=2*precision_sc*recall_sc*(precision_sc+recall_sc)
    except:
        f1_sc=0
    try:
        mcc_sc=((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    except:
        mcc_sc=0
    return specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc

def print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold):
    print('------------------------------------------------------------------------')
    print('Scores per fold')
    for i in range(0, len(acc_per_fold)):
      print('------------------------------------------------------------------------')
      print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}')
      print(f'            - Precision: {precision_per_fold[i]} - Recall: {recall_per_fold[i]}')
    print('------------------------------------------------------------------------')

def print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
                                   precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
                                   auc_per_fold,aupr_per_fold):
    print('Average scores for all folds:')
    
    print(f'>> Loss: {np.mean(loss_per_fold)} (+- {np.std(loss_per_fold)})')
    print(f'>> MSE: {np.mean(mse_per_fold)} (+- {np.std(mse_per_fold)})')
    print(f'>> Sensitivity: {np.mean(recall_per_fold)} (+- {np.std(recall_per_fold)})')
    print(f'>> Specivity: {np.mean(specivity_per_fold)} (+- {np.std(specivity_per_fold)})')
    print(f'>> Precision: {np.mean(precision_per_fold)} (+- {np.std(precision_per_fold)})')
    print(f'>> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'>> F1-Score: {np.mean(f1_per_fold)} (+- {np.std(f1_per_fold)})')
    print(f'>> MCC: {np.mean(mcc_per_fold)} (+- {np.std(mcc_per_fold)})')
    print(f'>> AUROC: {np.mean(auc_per_fold)} (+- {np.std(auc_per_fold)})')
    print(f'>> AUPRC: {np.mean(aupr_per_fold)} (+- {np.std(aupr_per_fold)})')
    print('------------------------------------------------------------------------')

def write_model_parameters_GRU(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                           min_delta,restore_best_weights,gru_units,batch_size,epochs,callbacks,
                           classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                           training_dataset,train,validation,monitor,mode,metric_name):
    with open(mparams_file, 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.write("program name: "+prog_name+ ' \n')
        f.write("device used: "+device_t+ ' \n')
        f.write("optimizer: "+optimizer_name+ ' \n')
        f.write("loss: "+loss+ ' \n')
        f.write("metrics: "+ str(metrics)+ ' \n')
        f.write("patience: "+ str(patience)+ ' \n')
        f.write("min_delta: "+str(min_delta)+ ' \n')
        f.write("restore_best_weights: "+str(restore_best_weights)+ ' \n')
        f.write("gru_unit: "+str(gru_units)+ ' \n')
        # f.write("validation_split: "+str(validation_split)+ ' \n')
        f.write("batch_size: "+str(batch_size)+ ' \n')
        # f.write("steps_per_epoch: "+str(steps_per_epoch)+ ' \n')
        f.write("epochs: "+str(epochs)+ ' \n')
        f.write("callbacks: "+str(callbacks)+ ' \n')
        f.write("classWeight: "+str(classWeight)+ ' \n')
        f.write("shuffle: "+str(shuffle)+ ' \n')
        f.write("monitor,mode,metric_name: "+monitor+' '+mode+' '+metric_name+ ' \n')
        f.write("Arguments: "+str(vars(args))+ ' \n')
        f.write("Scaling: "+str(scale_data)+ ' \n')
        f.write("Scaler: "+str(scaler)+ ' \n')
        f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ ' \n')
        f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ ' \n')
        f.write("Training dataset: "+training_dataset+ ' \n')
        f.write("The number of columns of training dataset w/o label: "+str(train.shape[1])+ ' \n')
        f.write("The size of sliding windows : "+str(train.shape[2])+ ' \n')
        f.write("The length of training dataset after splitting: "+str(train.shape[0])+ ' \n')
        f.write("The length of validation dataset: "+str(len(validation))+ ' \n')
        f.close()

def write_model_parameters_ML(mparams_file,model,prog_name,device_t,
                           classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                           training_dataset,train,validation):
    with open(mparams_file, 'w+') as f:
        # if hyperparams==True: 
        #     f.write("Best parameters: "+str(model.best_params_)+ '\n')
        #     f.write("Accuracy: "+str(model.best_score_)+ '\n')
        f.write("program name: "+prog_name+ ' \n')
        f.write("device used: "+device_t+ ' \n')
        f.write("Arguments: "+str(vars(args))+ '\n')
        # f.write("Default classifier: "+str(default_clsf)+ '\n')
        f.write("Scaling: "+str(scale_data)+ '\n')
        f.write("Scaler: "+str(scaler)+ '\n')
        f.write("sqrt of weights: "+str(sqrtw)+ '\n')
        f.write("sample_weight: "+str(sampw)+ '\n')
        f.write("The number of trees: "+str(model.tree_count_)+ '\n')
        f.write("Feature Importance: "+str(model.feature_importances_)+ '\n')
        f.write("Label of classes: "+str(model.classes_)+ '\n')
        f.write("The best score for validation: "+str(model.best_score_)+ '\n')
        f.write("The best iteration for validation: "+str(model.best_iteration_)+ '\n')
        f.write("Model parameters: "+str(model.get_params())+ '\n')
        f.write("Model parameters: "+str(model.get_all_params())+ '\n')
        f.write("The length of training dataset: "+str(len(train)+len(validation))+ '\n')
        if scale_data==True:
           f.write("The number of columns of training dataset w/o label: "+str(len(train[0][:]))+ '\n')
        else:
           f.write("The number of columns of training dataset w/o label: "+str(len(train.columns))+ '\n')
        f.write("The length of training dataset after splitting: "+str(len(train))+ '\n')
        f.write("The length of validation dataset: "+str(len(validation))+ '\n')
        f.close()

def write_model_parameters_ML_lgbm(mparams_file,model,prog_name,device_t,
                           classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                           training_dataset,train,validation):
    with open(mparams_file, 'w+') as f:
        # if hyperparams==True: 
        #     f.write("Best parameters: "+str(model.best_params_)+ '\n')
        #     f.write("Accuracy: "+str(model.best_score_)+ '\n')
        f.write("program name: "+prog_name+ ' \n')
        f.write("device used: "+device_t+ ' \n')
        f.write("Arguments: "+str(vars(args))+ '\n')
        # f.write("Default classifier: "+str(default_clsf)+ '\n')
        f.write("Scaling: "+str(scale_data)+ '\n')
        f.write("Scaler: "+str(scaler)+ '\n')
        f.write("sqrt of weights: "+str(sqrtw)+ '\n')
        f.write("sample_weight: "+str(sampw)+ '\n')
        # f.write("The number of estimators: "+str(model.n_estimators_)+ '\n')
        f.write("Feature Importance: "+str(model.feature_importances_)+ '\n')
        f.write("Label of classes: "+str(model.classes_)+ '\n')
        f.write("The best score for validation: "+str(model.best_score_)+ '\n')
        f.write("The best iteration for validation: "+str(model.best_iteration_)+ '\n')
        f.write("Model parameters: "+str(model.get_params())+ '\n')
        # f.write("Model parameters: "+str(model.get_all_params())+ '\n')
        # if novald==False: f.write("Evaluation results: "+str(model.evals_result_)+ '\n')
        f.write("The length of training dataset: "+str(len(train)+len(validation))+ '\n')
        if scale_data==True:
           f.write("The number of columns of training dataset w/o label: "+str(len(train[0][:]))+ '\n')
        else:
           f.write("The number of columns of training dataset w/o label: "+str(len(train.columns))+ '\n')
        f.write("The length of training dataset after splitting: "+str(len(train))+ '\n')
        f.write("The length of validation dataset: "+str(len(validation))+ '\n')
        f.close()
        
def write_model_parameters_ML_xgb(mparams_file,model,prog_name,device_t,
                           classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                           training_dataset,train,validation):
    with open(mparams_file, 'w+') as f:
        # if hyperparams==True: 
        #     f.write("Best parameters: "+str(model.best_params_)+ '\n')
        #     f.write("Accuracy: "+str(model.best_score_)+ '\n')
        f.write("program name: "+prog_name+ ' \n')
        f.write("device used: "+device_t+ ' \n')
        f.write("Arguments: "+str(vars(args))+ '\n')
        # f.write("Default classifier: "+str(default_clsf)+ '\n')
        f.write("Scaling: "+str(scale_data)+ '\n')
        f.write("Scaler: "+str(scaler)+ '\n')
        f.write("sqrt of weights: "+str(sqrtw)+ '\n')
        f.write("sample_weight: "+str(sampw)+ '\n')
        # f.write("The number of estimators: "+str(model.n_estimators_)+ '\n')
        f.write("Feature Importance: "+str(model.feature_importances_)+ '\n')
        f.write("Label of classes: "+str(model.classes_)+ '\n')
        # f.write("The best score for validation: "+str(model.best_score_)+ '\n')
        # f.write("The best iteration for validation: "+str(model.best_iteration_)+ '\n')
        f.write("Model parameters: "+str(model.get_params())+ '\n')
        # f.write("Model parameters: "+str(model.get_all_params())+ '\n')
        if novald==False: f.write("Evaluation results: "+str(model.evals_result())+ '\n')
        f.write("The length of training dataset: "+str(len(train)+len(validation))+ '\n')
        if scale_data==True:
           f.write("The number of columns of training dataset w/o label: "+str(len(train[0][:]))+ '\n')
        else:
           f.write("The number of columns of training dataset w/o label: "+str(len(train.columns))+ '\n')
        f.write("The length of training dataset after splitting: "+str(len(train))+ '\n')
        f.write("The length of validation dataset: "+str(len(validation))+ '\n')
        f.close()

def write_model_parameters_CNN(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                           min_delta,restore_best_weights,batch_size,epochs,callbacks,
                           classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                           training_dataset,train,validation,strides,kernel_size,n_filter,max_pool,
                           monitor,mode,metric_name):
    with open(mparams_file, 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.write("program name: "+prog_name+ ' \n')
        f.write("device used: "+device_t+ ' \n')
        f.write("optimizer: "+optimizer_name+ ' \n')
        f.write("loss: "+loss+ '\n')
        f.write("metrics: "+ str(metrics)+ ' \n')
        f.write("patience: "+ str(patience)+ ' \n')
        f.write("min_delta: "+str(min_delta)+ ' \n')
        f.write("restore_best_weights: "+str(restore_best_weights)+ ' \n')
        f.write("strides: "+str(strides)+ ' \n')
        f.write("kernel_size: "+str(kernel_size)+ ' \n')
        f.write("n_filter: "+str(n_filter)+ ' \n')
        f.write("max_pool: "+str(max_pool)+ ' \n')
        # f.write("validation_split: "+str(validation_split)+ ' \n')
        f.write("batch_size: "+str(batch_size)+ ' \n')
        # f.write("steps_per_epoch: "+str(steps_per_epoch)+ ' \n')
        f.write("epochs: "+str(epochs)+ ' \n')
        f.write("callbacks: "+str(callbacks)+ ' \n')
        f.write("classWeight: "+str(classWeight)+ ' \n')
        f.write("shuffle: "+str(shuffle)+ ' \n')
        f.write("monitor,mode,metric_name: "+monitor+' '+mode+' '+metric_name+ ' \n')
        f.write("Arguments: "+str(vars(args))+ ' \n')
        f.write("Scaling: "+str(scale_data)+ ' \n')
        f.write("Scaler: "+str(scaler)+ ' \n')
        f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ ' \n')
        f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ ' \n')
        f.write("Training dataset: "+training_dataset+ ' \n')
        f.write("The number of columns of training dataset w/o label: "+str(train.shape[1])+ ' \n')
        f.write("The size of sliding windows : "+str(train.shape[2])+ ' \n')
        f.write("The length of training dataset after splitting: "+str(train.shape[0])+ ' \n')
        f.write("The length of validation dataset: "+str(len(validation))+ ' \n')
        f.close()

def write_model_parameters_Ensb(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                           min_delta,restore_best_weights,batch_size,epochs,callbacks,
                           classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                           training_dataset,train,validation,monitor,mode,metric_name,start_input=" "):
    with open(mparams_file, 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.write("program name: "+prog_name+ ' \n')
        f.write("device used: "+device_t+ ' \n')
        f.write("optimizer: "+optimizer_name+ ' \n')
        f.write("loss: "+loss+ '\n')
        f.write("metrics: "+ str(metrics)+ ' \n')
        f.write("patience: "+ str(patience)+ ' \n')
        f.write("min_delta: "+str(min_delta)+ ' \n')
        f.write("restore_best_weights: "+str(restore_best_weights)+ ' \n')
        # f.write("validation_split: "+str(validation_split)+ ' \n')
        f.write("batch_size: "+str(batch_size)+ ' \n')
        # f.write("steps_per_epoch: "+str(steps_per_epoch)+ ' \n')
        f.write("epochs: "+str(epochs)+ ' \n')
        f.write("callbacks: "+str(callbacks)+ ' \n')
        f.write("classWeight: "+str(classWeight)+ ' \n')
        f.write("shuffle: "+str(shuffle)+ ' \n')
        f.write("start input: "+str(start_input)+ ' \n')
        f.write("monitor,mode,metric_name: "+monitor+' '+mode+' '+metric_name+ ' \n')
        f.write("Arguments: "+str(vars(args))+ ' \n')
        f.write("Scaling: "+str(scale_data)+ ' \n')
        f.write("Scaler: "+str(scaler)+ ' \n')
        f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ ' \n')
        f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ ' \n')
        f.write("Training dataset: "+training_dataset+ ' \n')
        f.write("The number of columns of training dataset w/o label: "+str(train.shape[1])+ ' \n')
        f.write("The size of sliding windows : "+str(train.shape[2])+ ' \n')
        f.write("The length of training dataset after splitting: "+str(train.shape[0])+ ' \n')
        f.write("The length of validation dataset: "+str(len(validation))+ ' \n')
        

def write_model_parameters_LSTMs2s(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                           min_delta,restore_best_weights,lstm_units,batch_size,epochs,callbacks,
                           classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                           training_dataset,train,validation,start_input,monitor,mode,metric_name):
    with open(mparams_file, 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.write("program name: "+prog_name+ ' \n')
        f.write("device used: "+device_t+ ' \n')
        f.write("optimizer: "+optimizer_name+ '\ n')
        f.write("loss: "+loss+ ' \n')
        f.write("metrics: "+ str(metrics)+ ' \n')
        f.write("patience: "+ str(patience)+ ' \n')
        f.write("min_delta: "+str(min_delta)+ ' \n')
        f.write("restore_best_weights: "+str(restore_best_weights)+ ' \n')
        f.write("lstm_unit: "+str(lstm_units)+ ' \n')
        # f.write("validation_split: "+str(validation_split)+ ' \n')
        f.write("batch_size: "+str(batch_size)+ ' \n')
        # f.write("steps_per_epoch: "+str(steps_per_epoch)+ ' \n')
        f.write("epochs: "+str(epochs)+ ' \n')
        f.write("callbacks: "+str(callbacks)+ ' \n')
        f.write("classWeight: "+str(classWeight)+ ' \n')
        f.write("shuffle: "+str(shuffle)+ ' \n')
        f.write("start input: "+str(start_input)+ ' \n')
        f.write("monitor,mode,metric_name: "+monitor+' '+mode+' '+metric_name+ ' \n')
        f.write("Arguments: "+str(vars(args))+ ' \n')
        f.write("Scaling: "+str(scale_data)+ ' \n')
        f.write("Scaler: "+str(scaler)+ ' \n')
        f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ ' \n')
        f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ ' \n')
        f.write("Training dataset: "+training_dataset+ ' \n')
        f.write("The number of columns of training dataset w/o label: "+str(train.shape[1])+ ' \n')
        f.write("The size of sliding windows : "+str(train.shape[2])+ ' \n')
        f.write("The length of training dataset after splitting: "+str(train.shape[0])+ ' \n')
        f.write("The length of validation dataset: "+str(len(validation))+ ' \n')
        f.close()

def write_model_parameters_GRUs2s(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                           min_delta,restore_best_weights,gru_units,batch_size,epochs,callbacks,
                           classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                           training_dataset,train,validation,start_input,monitor,mode,metric_name):
    with open(mparams_file, 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.write("program name: "+prog_name+ ' \n')
        f.write("device used: "+device_t+ ' \n')
        f.write("optimizer: "+optimizer_name+ '\ n')
        f.write("loss: "+loss+ ' \n')
        f.write("metrics: "+ str(metrics)+ ' \n')
        f.write("patience: "+ str(patience)+ ' \n')
        f.write("min_delta: "+str(min_delta)+ ' \n')
        f.write("restore_best_weights: "+str(restore_best_weights)+ ' \n')
        f.write("gru_unit: "+str(gru_units)+ ' \n')
        # f.write("validation_split: "+str(validation_split)+ ' \n')
        f.write("batch_size: "+str(batch_size)+ ' \n')
        # f.write("steps_per_epoch: "+str(steps_per_epoch)+ ' \n')
        f.write("epochs: "+str(epochs)+ ' \n')
        f.write("callbacks: "+str(callbacks)+ ' \n')
        f.write("classWeight: "+str(classWeight)+ ' \n')
        f.write("shuffle: "+str(shuffle)+ ' \n')
        f.write("start input: "+str(start_input)+ ' \n')
        f.write("monitor,mode,metric_name: "+monitor+' '+mode+' '+metric_name+ ' \n')
        f.write("Arguments: "+str(vars(args))+ ' \n')
        f.write("Scaling: "+str(scale_data)+ ' \n')
        f.write("Scaler: "+str(scaler)+ ' \n')
        f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ ' \n')
        f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ ' \n')
        f.write("Training dataset: "+training_dataset+ ' \n')
        f.write("The number of columns of training dataset w/o label: "+str(train.shape[1])+ ' \n')
        f.write("The size of sliding windows : "+str(train.shape[2])+ ' \n')
        f.write("The length of training dataset after splitting: "+str(train.shape[0])+ ' \n')
        f.write("The length of validation dataset: "+str(len(validation))+ ' \n')
        f.close()

def data_info(undersampling,all_data):
    if undersampling==True:
        under_1_1_training=True
        under_1_1_valid=True
        under_lbl="_under"
    elif all_data==True:
        under_1_1_training=False
        under_1_1_valid=False
        under_lbl="_all"
    else:
        under_1_1_training=False
        under_1_1_valid=False
        under_lbl=""
    training_dataset="df_alldataW_all.csv"
    return under_1_1_training, under_1_1_valid, under_lbl, training_dataset

def metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
                       loss_per_fold,mse_per_fold,recall_per_fold,
                       specivity_per_fold,precision_per_fold,acc_per_fold,
                       f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold):
    precision_per_fold.append(precision_sc)
    recall_per_fold.append(recall_sc)
    acc_per_fold.append(scores[model.metrics_names.index('binary_accuracy')])
    loss_per_fold.append(scores[model.metrics_names.index('loss')])
    try:
        mse_per_fold.append(scores[model.metrics_names.index('mean_squared_error')])
    except:
        mse_per_fold.append(scores[model.metrics_names.index('mse')])
    auc_per_fold.append(scores[model.metrics_names.index('auroc')])
    aupr_per_fold.append(scores[model.metrics_names.index('auprc')])
    f1_per_fold.append(f1_sc)
    specivity_per_fold.append(specivity_sc)
    mcc_per_fold.append(mcc_sc)
    return loss_per_fold,mse_per_fold,recall_per_fold, \
        specivity_per_fold,precision_per_fold,acc_per_fold, \
        f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold


def append_model_parameters_testsets(mparams_file, X_testsets):
    with open(mparams_file, 'a') as f:
        for tset_name,Xy in X_testsets.items():
            try:
                f.write("The shape of test dataset "+tset_name[2:]+": "+str(Xy[0].shape)+ ' \n')
            except:
                f.write("The shape of test dataset "+tset_name[2:]+":  "+ '\n')
            f.write("The length of test dataset "+tset_name[2:]+": "+str(len(Xy[0]))+ ' \n')
        f.close()
        
def append_thresholds_testsets(mparams_file, threshold_dict):
    with open(mparams_file, 'a') as f:
        f.write("Thresholds: "+str(threshold_dict)+ ' \n')
        f.close()

def reshape2Dto3D(X):
    return np.reshape(X, (X.shape[0],X.shape[1],X.shape[2],1))

def reshape_datasets(Xy_datasets):
    for X in Xy_datasets[::2]:
        if len(X)!=0:
           X = reshape2Dto3D(X) 
    return Xy_datasets

def print_results_folds(metric_name,mn_mtr_ls):
    print("Mean values of "+metric_name+"s: ",mn_mtr_ls)
    print("The fold (max mean "+metric_name+"): ",mn_mtr_ls.index(max(mn_mtr_ls))+1)
    
def test_results_LSTMs2s(model,X_testsets,test_thresholds,start_input,model_dir,
                         metrics_results,scale_data,scaler,clsf_model):
    for tset_name, Xy in X_testsets.items():
        X,y=Xy[0],Xy[1]
        print("Results for Dtestset "+tset_name[2:]+" \n")
        decoder_input=np.full((y.shape[0],), start_input)
        decoder_input=np.reshape(decoder_input,(decoder_input.shape[0],1,1))
        # decoder_input=np.array(np.random.choice([0, 1], size=(y.shape[0]),p=[0.9,0.1]))
        # print(sum(decoder_input),sum(decoder_input)/len(decoder_input)*100)
        # decoder_input=np.random.uniform(size=y.shape[0])
        # enc_states=encoder_model.predict(X)
        testset_prediction_eval(model,[X,decoder_input],y,model_dir,
                                metrics_results,tset_name,scale_data,
                                scaler,clsf_model,test_thresholds[tset_name])
        
def test_results_Ensbs2s(model,X_testsets,num_inputs,test_thresholds,start_input,model_dir,
                         metrics_results,scale_data,scaler,clsf_model):
    for tset_name, Xy in X_testsets.items():
        X,y=Xy[0],Xy[1]
        print("Results for Dtestset "+tset_name[2:]+" \n")
        decoder_input=np.full((y.shape[0],), start_input)
        # enc_states=encoder_model.predict(X)
        testset_prediction_eval(model,[X,decoder_input]*num_inputs,y,model_dir,
                                metrics_results,tset_name,scale_data,
                                scaler,clsf_model,test_thresholds[tset_name])

def test_results_ML(model,X_testsets,test_thresholds,model_dir,
                         metrics_results,scale_data,scaler,clsf_model):
    for tset_name, Xy in X_testsets.items():
        X,y=Xy[0],Xy[1]
        print("Results for Dtestset "+tset_name[2:]+" \n")
        testset_prediction_eval(model,X,y,model_dir,
                                metrics_results,tset_name,scale_data,
                                scaler,clsf_model,test_thresholds[tset_name])
        
def test_results_ML2(model,X_testsets,test_thresholds,model_dir,
                         metrics_results,scale_data,scaler,clsf_model):
    for tset_name, Xy in X_testsets.items():
        X,y=Xy[0],Xy[1]
        print("Results for Dtestset "+tset_name[2:]+" \n")
        testset_prediction_evalML(model,X,y,model_dir,
                                metrics_results,tset_name,scale_data,
                                scaler,clsf_model,test_thresholds[tset_name])

def test_results_LSTMs2s_encdec(encoder_model,decoder_model,X_testsets,test_thresholds,start_input,model_dir,
                         metrics_results,scale_data,scaler,clsf_model):
    for tset_name, Xy in X_testsets.items():
        X,y=Xy[0],Xy[1]
        print("Results for Dtestset "+tset_name[2:]+" \n")
        decoder_input=np.full((y.shape[0],), start_input)
        enc_states=encoder_model.predict(X)
        testset_prediction_eval(decoder_model,[decoder_input]+enc_states,y,model_dir,
                                metrics_results,tset_name,scale_data,
                                scaler,clsf_model,test_thresholds[tset_name])
def test_results_Ensb(model,X_testsets,num_inputs_rc,num_inputs_sa,test_thresholds,start_input,model_dir,
                         metrics_results,scale_data,scaler,clsf_model):
    for tset_name, Xy in X_testsets.items():
        X,y=Xy[0],Xy[1]
        print("Results for Dtestset "+tset_name[2:]+" \n")
        decoder_input=np.full((y.shape[0],), start_input)
        # enc_states=encoder_model.predict(X)
        testset_prediction_eval(model,[X]*num_inputs_rc+[X,decoder_input]*num_inputs_sa,y,model_dir,
                                metrics_results,tset_name,scale_data,
                                scaler,clsf_model,test_thresholds[tset_name])
        
def test_results_Ensb2(model,X_testsets,test_thresholds,model_dir,
                         metrics_results,scale_data,scaler,clsf_model):
    for tset_name, Xy in X_testsets.items():
        X,y=Xy[0],Xy[1]
        print("Results for Dtestset "+tset_name[2:]+" \n")
        testset_prediction_eval(model,X,y,model_dir,
                                metrics_results,tset_name,scale_data,
                                scaler,clsf_model,test_thresholds[tset_name])
        
def test_results_Ensb3(model,X_testsets,test_thresholds,model_dir,
                         metrics_results,scale_data,scaler,clsf_model):
    for tset_name, Xy in X_testsets.items():
        X,y=Xy[0],Xy[1]
        print("Results for Dtestset "+tset_name[2:]+" \n")
        testset_prediction_eval(model,[X]*2,y,model_dir,
                                metrics_results,tset_name,scale_data,
                                scaler,clsf_model,test_thresholds[tset_name])
        
def class_wght(classw,classwcomp,classWeight):
    if classw==True or classwcomp==True:
        return classWeight
    else: 
        return None

def device_type_control(model_dir,device_type):
    if tf.config.list_physical_devices('GPU'):
        device_type="GPU"
        print("GPUs available: ")
        print(tf.config.experimental.list_physical_devices('XLA_GPU'))
        print(tf.config.experimental.list_physical_devices('GPU'))
    print("Devices List: ")
    print(device_lib.list_local_devices())
    orig_stdout = sys.stdout
    f = open(model_dir+"/devices.txt", 'w+')
    sys.stdout = f
    for device in device_lib.list_local_devices():
        print(device)
    sys.stdout = orig_stdout
    f.close()
    return device_type

def vald_data_callback(all_data,novald,X_valid,y_valid,callbacks):
    if all_data==True and novald==True:
        validation_data=None
        callbacks=None
    else:
        validation_data=(X_valid,y_valid)
        callbacks=callbacks
    return validation_data, callbacks

def create_result_folder(results_dir,shape):
    results_dir=results_dir+"_"+str(shape[1])+"_"+str(shape[2])
    if not os.path.exists(results_dir): os.mkdir(results_dir)
    return results_dir

def create_result_folder2(results_dir,wsize,X_train):
    try:
        featr=str(len(X_train[0][:]))
    except:
        if len(wsize)!=0: 
            featr=str(int(len(X_train.columns)/int(wsize.split("_")[1])))
        else:
            featr=str(len(X_train.columns))
    if len(wsize)!=0:
        results_dir=results_dir+"_"+wsize.split("_")[1]+"_"+featr
    else:
        results_dir=results_dir+"_"+featr
    if not os.path.exists(results_dir): os.mkdir(results_dir)
    return results_dir

def kfoldRNN(results_dir,n_folds):
    under_1_1_training, under_1_1_valid, under_lbl, training_dataset = data_info(undersampling,all_data)
    Xy_datasets = load_data(under_lbl)
    X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
    results_dir = create_result_folder(results_dir,X_train.shape[:])
    
    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold , auc_per_fold , loss_per_fold = [], [], []
    precision_per_fold, recall_per_fold, aupr_per_fold=[], [], []
    f1_per_fold, specivity_per_fold, mcc_per_fold=[], [], []
    mse_per_fold, mn_mtr_ls=[], []
    scaler = StandardScaler()
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        if int(fld_no)!=-1 and fold_no!=int(fld_no): 
            fold_no+=1
            continue
        if all_data==True:
            train, validation = X_train[train_index], X_train[validation_index]
            target_train, target_val = y_train[train_index], y_train[validation_index]
        else:
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid
            
        if scale_data==True:
            tests=[validation]+Xy_datasets[4::2] if pdbsets==True else [validation]+Xy_datasets[10::2]
            train, tests, scaler = scale_datasets(train,tests,scaler_type)

        print("\nFold ",fold_no)
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        
        tf.keras.backend.clear_session()
        ### PARAMETERS ###
        optimizer_name='adam'
        learning_rate=0.002
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss='binary_crossentropy'
        metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000,name='auprc'),
                 tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000,name='auroc'),
                 'binary_accuracy','mse','Precision','Recall',
                 'TrueNegatives','FalsePositives',
                 'FalseNegatives','TruePositives']
        patience=4
        min_delta=1e-5
        restore_best_weights=True
        validation_split=0.2
        batch_size=1024
        steps_per_epoch=500
        epochs=9
        padding='same'
        gru_units=64
        dense_units=64
        dropout_rate=0.3
        clsf_model="RNN"
        tag=clsf_model.lower()
        device_type="CPU"
        prog_name=sys.argv[0]
        shuffle=True
        monitor,mode,metric_name="val_auprc",'max',"AUPRC"
        input_shape=train.shape[1:]
        class_num=np.unique(target_train)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=restore_best_weights,
            mode=mode
        )
        callbacks=[early_stopping]
        classWeight = classWeight_comp(class_num,target_train,sqrtw,classwcomp)
        print(classWeight)
        sample_weight = sampleWeight_comp(classWeight,target_train,sampw)
        ### =========== ###
        
        ### MODEL ###
        model=build_gru_model2(input_shape,gru_units,dense_units,dropout_rate,
                              optimizer,loss,metrics,tag,fold_no)
        model.summary()

        # Define checkpoint callback
        model_dir=results_dir+'/model_rnn_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=model_dir+'/model_graph_gru.png')
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
            early_stopping
            # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True, mode='max')
        ]
        
        device_t="/device:"+device_type_control(model_dir,device_type)+":0"
        
        classWeight = class_wght(classw,classwcomp,classWeight)
        print(classWeight)
        
        validation_data, callbacks = vald_data_callback(all_data,novald,
                                                        validation,target_val,callbacks_check)
        ### FIT MODEL ###
        with tf.device(device_t):
            history = model.fit(
                train, target_train, 
                epochs=epochs, 
                validation_data=validation_data,
                callbacks=callbacks,
                batch_size=batch_size,
                class_weight=classWeight,
                sample_weight=sample_weight,
                shuffle=shuffle
            )
        
        # Save model
        model.save(model_dir+'/model.tf')
        model.save_weights(model_dir+"/weights.tf")
        
        probas = model.predict(validation).ravel()
        plot_probas_valid(probas,model_dir)
        plot_distrb_pred_valid(probas,target_val,model_dir)
        labels = (probas >= 0.5).astype(int)
        y_pred = np.squeeze(np.asarray(labels))
        print(classification_report(target_val, y_pred))
        print(confusion_matrix(target_val,y_pred))
        
        scores = model.evaluate(validation, target_val, verbose=0)
        print_validation_results_fold(model, scores, fold_no)
        
        # print(model.metrics_names)
        
        specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc = compute_metrics(model,scores)
        
        loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold, \
        precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold, \
        auc_per_fold,aupr_per_fold = \
        metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
                               loss_per_fold,mse_per_fold,recall_per_fold,
                               specivity_per_fold,precision_per_fold,acc_per_fold,
                               f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold)
        fold_no = fold_no + 1
        
        print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold)
        
        print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
                                           precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
                                           auc_per_fold,aupr_per_fold)
        
        mparams_file=model_dir+"/modelparams.txt"
        
        write_model_parameters_GRU(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                                   min_delta,restore_best_weights,gru_units,batch_size,epochs,callbacks,
                                   classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                                   training_dataset,train,validation,monitor,mode,metric_name)
        
        plot_model_history(history,model_dir)
        del train,validation,target_train,target_val
        gc.collect()
        metrics_results=[]
                
        test_tags,tsets_thrshlds_dict=test_tags_thresholds(pdbsets)
        X_testsets=select_testsets(pdbsets,Xy_datasets)
        test_results_ML(model,X_testsets,tsets_thrshlds_dict,model_dir,
                             metrics_results,scale_data,scaler,clsf_model)
        
        append_model_parameters_testsets(mparams_file, X_testsets)
        append_thresholds_testsets(mparams_file, tsets_thrshlds_dict)
        
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        
        test_mtr=[i+"_1" for i in test_tags]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')[metric_name].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        del model
        gc.collect()
        if all_data==False and novald==False and int(fld_no)!=-1: break
        # if fold_no==3: break
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print_results_folds(metric_name,mn_mtr_ls)
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/'+tag+'.dir', 'w') as f:
        f.write(results_dir+f'/model_{tag}_fold/fold_{fd_slct}/model.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)

def kfoldCNN(results_dir,n_folds):
    under_1_1_training, under_1_1_valid, under_lbl, training_dataset = data_info(undersampling,all_data)
    Xy_datasets = load_data(under_lbl)
    X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
    results_dir = create_result_folder(results_dir,X_train.shape[:])
    
    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold , auc_per_fold , loss_per_fold = [], [], []
    precision_per_fold, recall_per_fold, aupr_per_fold=[], [], []
    f1_per_fold, specivity_per_fold, mcc_per_fold=[], [], []
    mse_per_fold, mn_mtr_ls=[], []
    scaler = StandardScaler()
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        if int(fld_no)!=-1 and fold_no!=int(fld_no): 
            fold_no+=1
            continue
        if all_data==True:
            train, validation = X_train[train_index], X_train[validation_index]
            target_train, target_val = y_train[train_index], y_train[validation_index]
        else:
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid

        if scale_data==True:
            tests=[validation]+Xy_datasets[4::2] if pdbsets==True else [validation]+Xy_datasets[10::2]
            train, tests, scaler = scale_datasets(train,tests,scaler_type)
            
        print("\nFold ",fold_no)
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        
        tf.keras.backend.clear_session()
        ### PARAMETERS ###
        optimizer_name='adam'
        learning_rate=0.002
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss='binary_crossentropy'
        metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000,name='auprc'),
                 tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000,name='auroc'),
                 'binary_accuracy','mse','Precision','Recall',
                 'TrueNegatives','FalsePositives',
                 'FalseNegatives','TruePositives']
        patience=4
        min_delta=1e-5
        restore_best_weights=True
        validation_split=0.2
        batch_size=1024
        steps_per_epoch=500
        epochs=8
        padding='same'
        dense_units=64
        dropout_rate=0.3
        clsf_model="CNN"
        tag=clsf_model.lower()
        device_type="CPU"
        prog_name=sys.argv[0]
        strides=(1, 1) 
        kernel_size=5
        n_filter=48
        max_pool=(2,2)
        shuffle=True
        monitor,mode,metric_name="val_auprc",'max',"AUPRC"
        input_shape=train.shape[1:]
        class_num=np.unique(target_train)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=restore_best_weights,
            mode=mode
        )
        callbacks=[early_stopping]
        classWeight = classWeight_comp(class_num,target_train,sqrtw,classwcomp)
        print(classWeight)
        sample_weight = sampleWeight_comp(classWeight,target_train,sampw)
        ### =========== ###
        
        ### MODEL ###
        model=build_cnn_model(input_shape,n_filter,kernel_size,strides,padding,
                            dense_units,max_pool,dropout_rate,optimizer,loss,metrics,tag,fold_no)
        model.summary()

        # Define checkpoint callback
        model_dir=results_dir+'/model_cnn_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=model_dir+'/model_graph_cnn.png')
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
            early_stopping
            # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True, mode='max')
        ]
        device_t="/device:"+device_type_control(model_dir,device_type)+":0"
        
        classWeight = class_wght(classw,classwcomp,classWeight)
        print(classWeight)
        
        validation_data, callbacks = vald_data_callback(all_data,novald,
                                                        validation,target_val,callbacks_check)
        ### FIT MODEL ###
        with tf.device(device_t):
            history = model.fit(
                train, target_train, 
                epochs=epochs, 
                validation_data=validation_data,
                callbacks=callbacks,
                batch_size=batch_size,
                class_weight=classWeight,
                sample_weight=sample_weight,
                shuffle=shuffle
            )
        
        # Save model
        model.save(model_dir+'/model.tf')
        model.save_weights(model_dir+"/weights.tf")
        
        probas = model.predict(validation).ravel()
        plot_probas_valid(probas,model_dir)
        plot_distrb_pred_valid(probas,target_val,model_dir)
        labels = (probas >= 0.5).astype(int)
        y_pred = np.squeeze(np.asarray(labels))
        print(classification_report(target_val, y_pred))
        print(confusion_matrix(target_val,y_pred))
        
        scores = model.evaluate(validation, target_val, verbose=0)
        print_validation_results_fold(model, scores, fold_no)
        
        # print(model.metrics_names)
        
        specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc = compute_metrics(model,scores)
        
        loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold, \
        precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold, \
        auc_per_fold,aupr_per_fold = \
        metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
                               loss_per_fold,mse_per_fold,recall_per_fold,
                               specivity_per_fold,precision_per_fold,acc_per_fold,
                               f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold)
        fold_no = fold_no + 1
        
        print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold)
        
        print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
                                           precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
                                           auc_per_fold,aupr_per_fold)
        
        mparams_file=model_dir+"/modelparams.txt"

        write_model_parameters_CNN(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                                   min_delta,restore_best_weights,batch_size,epochs,callbacks,
                                   classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                                   training_dataset,train,validation,strides,kernel_size,n_filter,max_pool,
                                   monitor,mode,metric_name)
        
        plot_model_history(history,model_dir)
        del train,validation,target_train,target_val
        gc.collect()
        metrics_results=[]
                
        test_tags,tsets_thrshlds_dict=test_tags_thresholds(pdbsets)
        X_testsets=select_testsets(pdbsets,Xy_datasets)
        test_results_ML(model,X_testsets,tsets_thrshlds_dict,model_dir,
                             metrics_results,scale_data,scaler,clsf_model)
        
        append_model_parameters_testsets(mparams_file, X_testsets)
        append_thresholds_testsets(mparams_file, tsets_thrshlds_dict)
        
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        
        test_mtr=[i+"_1" for i in test_tags]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')[metric_name].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        del model
        gc.collect()
        if all_data==False and novald==False and int(fld_no)!=-1: break
        # if fold_no==3: break
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print_results_folds(metric_name,mn_mtr_ls)
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/'+tag+'.dir', 'w') as f:
        f.write(results_dir+f'/model_{tag}_fold/fold_{fd_slct}/model.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)
    
def kfoldEnsemble(results_dir,n_folds):
    under_1_1_training, under_1_1_valid, under_lbl, training_dataset = data_info(undersampling,all_data)
    Xy_datasets = load_data(under_lbl)
    X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]

    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold , auc_per_fold , loss_per_fold = [], [], []
    precision_per_fold, recall_per_fold, aupr_per_fold=[], [], []
    f1_per_fold, specivity_per_fold, mcc_per_fold=[], [], []
    mse_per_fold, mn_mtr_ls=[], []
    scaler = StandardScaler()
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        if fold_no<5: 
            fold_no+=1
            continue
        if all_data==True:
            train, validation = X_train[train_index], X_train[validation_index]
            target_train, target_val = y_train[train_index], y_train[validation_index]
        else:
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid

        if scale_data==True:
            tests=[validation]+Xy_datasets[4::2] if pdbsets==True else [validation]+Xy_datasets[10::2]
            train, tests, scaler = scale_datasets(train,tests,scaler_type)
        print("\nFold ",fold_no)
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        
        tf.keras.backend.clear_session()
        ### PARAMETERS ###
        optimizer_name='adam'
        learning_rate=0.002
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss='binary_crossentropy'
        metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000,name='auprc'),
                 tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000,name='auroc'),
                 'binary_accuracy','mse','Precision','Recall',
                 'TrueNegatives','FalsePositives',
                 'FalseNegatives','TruePositives']
        patience=3
        min_delta=1e-5
        restore_best_weights=True
        validation_split=0.2
        batch_size=1024
        steps_per_epoch=500
        epochs=5
        dense_units=64
        dropout_rate=0.5
        clsf_model="Ensb"
        tag=clsf_model.lower()
        device_type="CPU"
        prog_name=sys.argv[0]
        n_models_rnn=5
        n_models_cnn=5
        n_models_ls2satt=5
        n_models_ls2s=5
        shuffle=True
        start_input=-2
        monitor,mode,metric_name="val_auprc",'max',"AUPRC"
        class_num=np.unique(target_train)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=restore_best_weights,
            mode=mode
        )
        callbacks=[early_stopping]
        classWeight = classWeight_comp(class_num,target_train,sqrtw,classwcomp)
        print(classWeight)
        sample_weight = sampleWeight_comp(classWeight,target_train,sampw)
        ### =========== ###
        
        ### MODEL ###
        model = build_ensemble_model(results_dir,n_models_rnn,n_models_cnn,
                                  n_models_ls2satt,n_models_ls2s,dense_units,
                                 dropout_rate,optimizer,loss,metrics,fold_no)
        model.summary()
        
        train_decoder_input=np.full((target_train.shape[0],), start_input)
        valid_decoder_input=np.full((target_val.shape[0],), start_input)        
        
        # Define checkpoint callback
        model_dir=results_dir+'/model_ensb_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=model_dir+'/model_graph_ensb.png')
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
            early_stopping
            # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True, mode='max')
        ]
        num_inputs_rc=(n_models_rnn+n_models_cnn)
        num_inputs_sa=(n_models_ls2s+n_models_ls2satt)
        val_inputs=[validation]*num_inputs_rc+[validation,valid_decoder_input]*num_inputs_sa
        train_inputs =[train]*num_inputs_rc+[train,train_decoder_input]*num_inputs_sa
        
        device_t="/device:"+device_type_control(model_dir,device_type)+":0"
        
        classWeight = class_wght(classw,classwcomp,classWeight)
        print(classWeight)
        
        validation_data, callbacks = vald_data_callback(all_data,novald,
                                                        val_inputs,target_val,callbacks_check)
        ### FIT MODEL ###
        with tf.device(device_t):
            history = model.fit(
                train_inputs, target_train, 
                epochs=epochs, 
                validation_data=validation_data,
                callbacks=callbacks,
                batch_size=batch_size,
                class_weight=classWeight,
                sample_weight=sample_weight,
                shuffle=shuffle
            )
            
        # Save model
        model.save(model_dir+'/model.tf')
        model.save_weights(model_dir+"/weights.tf")
        
        # probas = model.predict(val_inputs)

        # labels = (probas >= 0.5).astype(int)
        # y_pred = np.squeeze(np.asarray(labels))
        # print(classification_report(target_val, y_pred))
        # print(confusion_matrix(target_val,y_pred))
        scores = model.evaluate(val_inputs, target_val, verbose=0)
        print_validation_results_fold(model, scores, fold_no)
        
        # print(model.metrics_names)
        
        specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc = compute_metrics(model,scores)
        
        loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold, \
        precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold, \
        auc_per_fold,aupr_per_fold = \
        metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
                                loss_per_fold,mse_per_fold,recall_per_fold,
                                specivity_per_fold,precision_per_fold,acc_per_fold,
                                f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold)
        fold_no = fold_no + 1
        
        print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold)
        
        print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
                                            precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
                                            auc_per_fold,aupr_per_fold)
        
        mparams_file=model_dir+"/modelparams.txt"
        
        write_model_parameters_Ensb(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                                   min_delta,restore_best_weights,batch_size,epochs,callbacks,
                                   classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                                   training_dataset,train,validation,monitor,mode,metric_name,start_input)
        
        plot_model_history(history,model_dir)
        
        metrics_results=[]
        
        test_tags,tsets_thrshlds_dict=test_tags_thresholds(pdbsets)
        X_testsets=select_testsets(pdbsets,Xy_datasets)
        test_results_Ensbs2s(model,X_testsets,num_inputs_rc,num_inputs_sa,tsets_thrshlds_dict,start_input,model_dir,
                             metrics_results,scale_data,scaler,clsf_model)
        
        append_model_parameters_testsets(mparams_file, X_testsets)
        append_thresholds_testsets(mparams_file, tsets_thrshlds_dict)
        
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        
        test_mtr=[i+"_1" for i in test_tags]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')[metric_name].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        if all_data==False and novald==False: break
        # if fold_no==3: break
        del train,validation,target_train,target_val,model, \
        val_inputs,train_inputs,train_decoder_input,valid_decoder_input
        gc.collect()
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print_results_folds(metric_name,mn_mtr_ls)
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/'+tag+'.dir', 'w') as f:
        f.write(results_dir+f'/model_{tag}_fold/fold_{fd_slct}/model.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)
    
def kfoldEnsemble2(results_dir,n_folds):
    under_1_1_training, under_1_1_valid, under_lbl, training_dataset = data_info(undersampling,all_data)
    Xy_datasets = load_data(under_lbl)
    X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]

    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold , auc_per_fold , loss_per_fold = [], [], []
    precision_per_fold, recall_per_fold, aupr_per_fold=[], [], []
    f1_per_fold, specivity_per_fold, mcc_per_fold=[], [], []
    mse_per_fold, mn_mtr_ls=[], []
    scaler = StandardScaler()
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        # if fold_no<5: 
        #     fold_no+=1
        #     continue
        if all_data==True:
            train, validation = X_train[train_index], X_train[validation_index]
            target_train, target_val = y_train[train_index], y_train[validation_index]
        else:
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid

        if scale_data==True:
            tests=[validation]+Xy_datasets[4::2] if pdbsets==True else [validation]+Xy_datasets[10::2]
            train, tests, scaler = scale_datasets(train,tests,scaler_type)
        print("\nFold ",fold_no)
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        
        tf.keras.backend.clear_session()
        ### PARAMETERS ###
        optimizer_name='adam'
        learning_rate=0.002
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss='binary_crossentropy'
        metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000,name='auprc'),
                 tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000,name='auroc'),
                 'binary_accuracy','mse','Precision','Recall',
                 'TrueNegatives','FalsePositives',
                 'FalseNegatives','TruePositives']
        patience=4
        min_delta=1e-5
        restore_best_weights=True
        validation_split=0.2
        batch_size=1024
        steps_per_epoch=500
        epochs=5
        dense_units=96
        dropout_rate=0.3
        clsf_model="Ensb"
        tag=clsf_model.lower()
        device_type="CPU"
        prog_name=sys.argv[0]
        shuffle=True
        start_input=-2
        input_shape=train.shape[1:]
        monitor,mode,metric_name="val_auprc",'max',"AUPRC"
        class_num=np.unique(target_train)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=restore_best_weights,
            mode=mode
        )
        callbacks=[early_stopping]
        classWeight = classWeight_comp(class_num,target_train,sqrtw,classwcomp)
        print(classWeight)
        sample_weight = sampleWeight_comp(classWeight,target_train,sampw)
        ### =========== ###
        
        ### MODEL ###
        model = build_ensemble_model_rc2(results_dir,input_shape,
                                  dense_units,dropout_rate,optimizer,loss,metrics,tag,fold_no)
        model.summary()
        
        # Define checkpoint callback
        model_dir=results_dir+'/model_ensb_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=model_dir+'/model_graph_ensb.png')
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
            early_stopping
            # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True, mode='max')
        ]
        device_t="/device:"+device_type_control(model_dir,device_type)+":0"
        
        classWeight = class_wght(classw,classwcomp,classWeight)
        print(classWeight)
        validation_data, callbacks = vald_data_callback(all_data,novald,
                                                        validation,target_val,callbacks_check)
        ### FIT MODEL ###
        with tf.device(device_t):
            history = model.fit(
                train, target_train, 
                epochs=epochs, 
                validation_data=validation_data,
                callbacks=callbacks,
                batch_size=batch_size,
                class_weight=classWeight,
                sample_weight=sample_weight,
                shuffle=shuffle
            )
            
        # Save model
        model.save(model_dir+'/model.tf')
        model.save_weights(model_dir+"/weights.tf")
        
        # probas = model.predict(val_inputs)

        # labels = (probas >= 0.5).astype(int)
        # y_pred = np.squeeze(np.asarray(labels))
        # print(classification_report(target_val, y_pred))
        # print(confusion_matrix(target_val,y_pred))
        scores = model.evaluate(validation, target_val, verbose=0)
        print_validation_results_fold(model, scores, fold_no)
        
        # print(model.metrics_names)
        
        specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc = compute_metrics(model,scores)
        
        loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold, \
        precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold, \
        auc_per_fold,aupr_per_fold = \
        metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
                                loss_per_fold,mse_per_fold,recall_per_fold,
                                specivity_per_fold,precision_per_fold,acc_per_fold,
                                f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold)
        fold_no = fold_no + 1
        
        print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold)
        
        print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
                                            precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
                                            auc_per_fold,aupr_per_fold)
        
        mparams_file=model_dir+"/modelparams.txt"
        
        write_model_parameters_Ensb(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                                   min_delta,restore_best_weights,batch_size,epochs,callbacks,
                                   classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                                   training_dataset,train,validation,monitor,mode,metric_name,start_input)
        
        plot_model_history(history,model_dir)
        
        metrics_results=[]
        
        test_tags,tsets_thrshlds_dict=test_tags_thresholds(pdbsets)
        X_testsets=select_testsets(pdbsets,Xy_datasets)
        test_results_Ensb2(model,X_testsets,tsets_thrshlds_dict,model_dir,
                             metrics_results,scale_data,scaler,clsf_model)
        
        append_model_parameters_testsets(mparams_file, X_testsets)
        append_thresholds_testsets(mparams_file, tsets_thrshlds_dict)
        
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        
        test_mtr=[i+"_1" for i in test_tags]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')[metric_name].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        del train,validation,target_train,target_val,model
        gc.collect()
        if all_data==False and novald==False: break
        break
        # if fold_no==3: break
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print_results_folds(metric_name,mn_mtr_ls)
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/'+tag+'.dir', 'w') as f:
        f.write(results_dir+f'/model_{tag}_fold/fold_{fd_slct}/model.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)

def kfoldEnsemble3(results_dir,n_folds):
    under_1_1_training, under_1_1_valid, under_lbl, training_dataset = data_info(undersampling,all_data)
    Xy_datasets = load_data(under_lbl)
    X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]

    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold , auc_per_fold , loss_per_fold = [], [], []
    precision_per_fold, recall_per_fold, aupr_per_fold=[], [], []
    f1_per_fold, specivity_per_fold, mcc_per_fold=[], [], []
    mse_per_fold, mn_mtr_ls=[], []
    scaler = StandardScaler()
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        # if fold_no<5: 
        #     fold_no+=1
        #     continue
        if all_data==True:
            train, validation = X_train[train_index], X_train[validation_index]
            target_train, target_val = y_train[train_index], y_train[validation_index]
        else:
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid

        if scale_data==True:
            tests=[validation]+Xy_datasets[4::2] if pdbsets==True else [validation]+Xy_datasets[10::2]
            train, tests, scaler = scale_datasets(train,tests,scaler_type)
        print("\nFold ",fold_no)
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        
        tf.keras.backend.clear_session()
        ### PARAMETERS ###
        optimizer_name='adam'
        learning_rate=0.002
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss='binary_crossentropy'
        metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000,name='auprc'),
                 tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000,name='auroc'),
                 'binary_accuracy','mse','Precision','Recall',
                 'TrueNegatives','FalsePositives',
                 'FalseNegatives','TruePositives']
        patience=4
        min_delta=1e-5
        restore_best_weights=True
        validation_split=0.2
        batch_size=1024
        steps_per_epoch=500
        epochs=5
        dense_units=96
        dropout_rate=0.3
        clsf_model="Ensb"
        tag=clsf_model.lower()
        device_type="CPU"
        prog_name=sys.argv[0]
        shuffle=True
        start_input=-2
        input_shape=train.shape[1:]
        monitor,mode,metric_name="val_auprc",'max',"AUPRC"
        class_num=np.unique(target_train)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=restore_best_weights,
            mode=mode
        )
        callbacks=[early_stopping]
        classWeight = classWeight_comp(class_num,target_train,sqrtw,classwcomp)
        print(classWeight)
        sample_weight = sampleWeight_comp(classWeight,target_train,sampw)
        ### =========== ###
        
        ### MODEL ###
        model = build_ensemble_model_rc3(results_dir,input_shape,
                                  dense_units,dropout_rate,optimizer,loss,metrics,tag,fold_no)
        model.summary()
        
        # Define checkpoint callback
        model_dir=results_dir+'/model_ensb_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=model_dir+'/model_graph_ensb.png')
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
            early_stopping
            # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True, mode='max')
        ]
        device_t="/device:"+device_type_control(model_dir,device_type)+":0"
        
        classWeight = class_wght(classw,classwcomp,classWeight)
        print(classWeight)
        validation_data, callbacks = vald_data_callback(all_data,novald,
                                                        [validation]*2,target_val,callbacks_check)
        ### FIT MODEL ###
        with tf.device(device_t):
            history = model.fit(
                [train]*2, target_train, 
                epochs=epochs, 
                validation_data=validation_data,
                callbacks=callbacks,
                batch_size=batch_size,
                class_weight=classWeight,
                sample_weight=sample_weight,
                shuffle=shuffle
            )
            
        # Save model
        model.save(model_dir+'/model.tf')
        model.save_weights(model_dir+"/weights.tf")
        
        # probas = model.predict(val_inputs)

        # labels = (probas >= 0.5).astype(int)
        # y_pred = np.squeeze(np.asarray(labels))
        # print(classification_report(target_val, y_pred))
        # print(confusion_matrix(target_val,y_pred))
        scores = model.evaluate([validation]*2, target_val, verbose=0)
        print_validation_results_fold(model, scores, fold_no)
        
        # print(model.metrics_names)
        
        specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc = compute_metrics(model,scores)
        
        loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold, \
        precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold, \
        auc_per_fold,aupr_per_fold = \
        metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
                                loss_per_fold,mse_per_fold,recall_per_fold,
                                specivity_per_fold,precision_per_fold,acc_per_fold,
                                f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold)
        fold_no = fold_no + 1
        
        print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold)
        
        print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
                                            precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
                                            auc_per_fold,aupr_per_fold)
        
        mparams_file=model_dir+"/modelparams.txt"
        
        write_model_parameters_Ensb(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                                   min_delta,restore_best_weights,batch_size,epochs,callbacks,
                                   classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                                   training_dataset,train,validation,monitor,mode,metric_name,start_input)
        
        plot_model_history(history,model_dir)
        
        metrics_results=[]
        
        test_tags,tsets_thrshlds_dict=test_tags_thresholds(pdbsets)
        X_testsets=select_testsets(pdbsets,Xy_datasets)
        test_results_Ensb3(model,X_testsets,tsets_thrshlds_dict,model_dir,
                             metrics_results,scale_data,scaler,clsf_model)
        
        append_model_parameters_testsets(mparams_file, X_testsets)
        append_thresholds_testsets(mparams_file, tsets_thrshlds_dict)
        
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        
        test_mtr=[i+"_1" for i in test_tags]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')[metric_name].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        del train,validation,target_train,target_val,model
        gc.collect()
        if all_data==False and novald==False: break
        break
        # if fold_no==3: break
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print_results_folds(metric_name,mn_mtr_ls)
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/'+tag+'.dir', 'w') as f:
        f.write(results_dir+f'/model_{tag}_fold/fold_{fd_slct}/model.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)

def kfoldEnsembles2s(results_dir,n_folds):
    under_1_1_training, under_1_1_valid, under_lbl, training_dataset = data_info(undersampling,all_data)
    Xy_datasets = load_data(under_lbl)
    X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]

    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold , auc_per_fold , loss_per_fold = [], [], []
    precision_per_fold, recall_per_fold, aupr_per_fold=[], [], []
    f1_per_fold, specivity_per_fold, mcc_per_fold=[], [], []
    mse_per_fold, mn_mtr_ls=[], []
    scaler = StandardScaler()
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        if fold_no<5: 
            fold_no+=1
            continue
        if all_data==True:
            train, validation = X_train[train_index], X_train[validation_index]
            target_train, target_val = y_train[train_index], y_train[validation_index]
        else:
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid

        if scale_data==True:
            tests=[validation]+Xy_datasets[4::2] if pdbsets==True else [validation]+Xy_datasets[10::2]
            train, tests, scaler = scale_datasets(train,tests,scaler_type)
        print("\nFold ",fold_no)
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        
        tf.keras.backend.clear_session()
        ### PARAMETERS ###
        optimizer_name='adam'
        learning_rate=0.002
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss='binary_crossentropy'
        metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000,name='auprc'),
                 tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000,name='auroc'),
                 'binary_accuracy','mse','Precision','Recall',
                 'TrueNegatives','FalsePositives',
                 'FalseNegatives','TruePositives']
        patience=3
        min_delta=1e-5
        restore_best_weights=True
        validation_split=0.2
        batch_size=512
        steps_per_epoch=500
        epochs=3
        dense_units=64
        dropout_rate=0.5
        clsf_model="Ensb"
        tag=clsf_model.lower()
        device_type="CPU"
        n_models_rnn=5
        n_models_cnn=5
        n_models_ls2s=5
        n_models_ls2satt=5
        prog_name=sys.argv[0]
        shuffle=True
        start_input=-2
        monitor,mode,metric_name="val_auprc",'max',"AUPRC"
        models_tags_num={"rnn":n_models_rnn,"cnn":n_models_cnn,
                         "lstms2s":n_models_ls2s,"lstms2satt":n_models_ls2satt}
        class_num=np.unique(target_train)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=restore_best_weights,
            mode=mode
        )
        callbacks=[early_stopping]
        classWeight = classWeight_comp(class_num,target_train,sqrtw,classwcomp)
        print(classWeight)
        sample_weight = sampleWeight_comp(classWeight,target_train,sampw)
        ### =========== ###
        
        ### MODEL ###
        model = build_ensemble_model(results_dir,models_tags_num,dense_units,
                                 dropout_rate,optimizer,loss,metrics,fold_no)
        model.summary()
        
        train_decoder_input=np.full((target_train.shape[0],), start_input)
        valid_decoder_input=np.full((target_val.shape[0],), start_input)        
        
        # Define checkpoint callback
        model_dir=results_dir+'/model_ensb_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=model_dir+'/model_graph_ensb.png')
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
            early_stopping
            # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True, mode='max')
        ]
        num_inputs=(n_models_ls2s+n_models_ls2satt)
        val_inputs=[validation,valid_decoder_input]*num_inputs
        train_inputs =[train,train_decoder_input]*num_inputs
        
        device_t="/device:"+device_type_control(model_dir,device_type)+":0"
        
        classWeight = class_wght(classw,classwcomp,classWeight)
        print(classWeight)
        
        validation_data, callbacks = vald_data_callback(all_data,novald,
                                                        val_inputs,target_val,callbacks_check)
        ### FIT MODEL ###
        with tf.device(device_t):
            history = model.fit(
                train_inputs, target_train, 
                epochs=epochs, 
                validation_data=validation_data,
                callbacks=callbacks,
                batch_size=batch_size,
                class_weight=classWeight,
                sample_weight=sample_weight,
                shuffle=shuffle
            )
            
        # Save model
        model.save(model_dir+'/model.tf')
        model.save_weights(model_dir+"/weights.tf")
        
        # probas = model.predict(val_inputs)

        # labels = (probas >= 0.5).astype(int)
        # y_pred = np.squeeze(np.asarray(labels))
        # print(classification_report(target_val, y_pred))
        # print(confusion_matrix(target_val,y_pred))
        scores = model.evaluate(val_inputs, target_val, verbose=0)
        print_validation_results_fold(model, scores, fold_no)
        
        # print(model.metrics_names)
        
        specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc = compute_metrics(model,scores)
        
        loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold, \
        precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold, \
        auc_per_fold,aupr_per_fold = \
        metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
                                loss_per_fold,mse_per_fold,recall_per_fold,
                                specivity_per_fold,precision_per_fold,acc_per_fold,
                                f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold)
        fold_no = fold_no + 1
        
        print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold)
        
        print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
                                            precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
                                            auc_per_fold,aupr_per_fold)
        
        mparams_file=model_dir+"/modelparams.txt"
        
        write_model_parameters_Ensb(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                                   min_delta,restore_best_weights,batch_size,epochs,callbacks,
                                   classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                                   training_dataset,train,validation,monitor,mode,metric_name,start_input)
        
        plot_model_history(history,model_dir)
        
        metrics_results=[]
        
        test_tags,tsets_thrshlds_dict=test_tags_thresholds(pdbsets)
        X_testsets=select_testsets(pdbsets,Xy_datasets)
        test_results_Ensbs2s(model,X_testsets,num_inputs,tsets_thrshlds_dict,start_input,model_dir,
                             metrics_results,scale_data,scaler,clsf_model)
        
        append_model_parameters_testsets(mparams_file, X_testsets)
        append_thresholds_testsets(mparams_file, tsets_thrshlds_dict)
        
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        
        test_mtr=[i+"_1" for i in test_tags]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')[metric_name].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        del train,validation,target_train,target_val,model, \
        val_inputs,train_inputs,train_decoder_input,valid_decoder_input
        gc.collect()
        if all_data==False and novald==False: break
        # if fold_no==3: break
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print_results_folds(metric_name,mn_mtr_ls)
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/'+tag+'.dir', 'w') as f:
        f.write(results_dir+f'/model_{tag}_fold/fold_{fd_slct}/model.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)

def kfoldEnsemble4(results_dir,n_folds):
    under_1_1_training, under_1_1_valid, under_lbl, training_dataset = data_info(undersampling,all_data)
    Xy_datasets = load_data(under_lbl)
    X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
    results_dir = create_result_folder(results_dir,X_train.shape[:])

    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold , auc_per_fold , loss_per_fold = [], [], []
    precision_per_fold, recall_per_fold, aupr_per_fold=[], [], []
    f1_per_fold, specivity_per_fold, mcc_per_fold=[], [], []
    mse_per_fold, mn_mtr_ls=[], []
    scaler = StandardScaler()
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        if int(fld_no)!=-1 and fold_no!=int(fld_no): 
            fold_no+=1
            continue
        if all_data==True:
            train, validation = X_train[train_index], X_train[validation_index]
            target_train, target_val = y_train[train_index], y_train[validation_index]
        else:
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid

        if scale_data==True:
            tests=[validation]+Xy_datasets[4::2] if pdbsets==True else [validation]+Xy_datasets[10::2]
            train, tests, scaler = scale_datasets(train,tests,scaler_type)
        print("\nFold ",fold_no)
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        
        tf.keras.backend.clear_session()
        ### PARAMETERS ###
        optimizer_name='adam'
        learning_rate=0.002
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss='binary_crossentropy'
        metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000,name='auprc'),
                 tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000,name='auroc'),
                 'binary_accuracy','mse','Precision','Recall',
                 'TrueNegatives','FalsePositives',
                 'FalseNegatives','TruePositives']
        patience=4
        min_delta=1e-5
        restore_best_weights=True
        validation_split=0.2
        batch_size=1024
        steps_per_epoch=500
        epochs=5
        dense_units=128
        dropout_rate=0.5
        clsf_model="Ensb"
        tag=clsf_model.lower()
        device_type="CPU"
        prog_name=sys.argv[0]
        shuffle=True
        start_input=-2
        input_shape=train.shape[1:]
        n_models_rnn=5
        n_models_cnn=5
        n_models_ls2s=5
        n_models_ls2satt=5
        monitor,mode,metric_name="val_auprc",'max',"AUPRC"
        models_tags_num={"rnn":n_models_rnn,"cnn":n_models_cnn,
                         "grus2s":n_models_ls2s,"grus2satt":n_models_ls2satt}
        class_num=np.unique(target_train)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=restore_best_weights,
            mode=mode
        )
        callbacks=[early_stopping]
        classWeight = classWeight_comp(class_num,target_train,sqrtw,classwcomp)
        print(classWeight)
        sample_weight = sampleWeight_comp(classWeight,target_train,sampw)
        ### =========== ###
        
        ### MODEL ###
        model = build_ensemble_model3(results_dir,models_tags_num,input_shape,dense_units,
                                 dropout_rate,optimizer,loss,metrics,tag,fold_no)
        model.summary()
        
        train_decoder_input=np.full((target_train.shape[0],), start_input)
        train_decoder_input=np.reshape(train_decoder_input,(train_decoder_input.shape[0],1,1))
        valid_decoder_input=np.full((target_val.shape[0],), start_input)
        valid_decoder_input=np.reshape(valid_decoder_input,(valid_decoder_input.shape[0],1,1)) 
        
        # Define checkpoint callback
        model_dir=results_dir+'/model_ensb_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=model_dir+'/model_graph_ensb.png')
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
            early_stopping
            # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True, mode='max')
        ]
        val_inputs=[validation,valid_decoder_input]
        train_inputs =[train,train_decoder_input]
        
        device_t="/device:"+device_type_control(model_dir,device_type)+":0"
        
        classWeight = class_wght(classw,classwcomp,classWeight)
        print(classWeight)
        validation_data, callbacks = vald_data_callback(all_data,novald,
                                                        val_inputs,target_val,callbacks_check)
        ### FIT MODEL ###
        with tf.device(device_t):
            history = model.fit(
                train_inputs, target_train, 
                epochs=epochs, 
                validation_data=validation_data,
                callbacks=callbacks,
                batch_size=batch_size,
                class_weight=classWeight,
                sample_weight=sample_weight,
                shuffle=shuffle
            )
            
        # Save model
        model.save(model_dir+'/model.tf')
        model.save_weights(model_dir+"/weights.tf")
        
        probas = model.predict(val_inputs).ravel()
        plot_probas_valid(probas,model_dir)
        plot_distrb_pred_valid(probas,target_val,model_dir)
        labels = (probas >= 0.5).astype(int)
        y_pred = np.squeeze(np.asarray(labels))
        print(classification_report(target_val, y_pred))
        print(confusion_matrix(target_val,y_pred))

        scores = model.evaluate(val_inputs, target_val, verbose=0)
        print_validation_results_fold(model, scores, fold_no)
        
        print(model.metrics_names)
        
        specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc = compute_metrics(model,scores)
        
        loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold, \
        precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold, \
        auc_per_fold,aupr_per_fold = \
        metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
                                loss_per_fold,mse_per_fold,recall_per_fold,
                                specivity_per_fold,precision_per_fold,acc_per_fold,
                                f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold)
        fold_no = fold_no + 1
        
        print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold)
        
        print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
                                            precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
                                            auc_per_fold,aupr_per_fold)
        
        mparams_file=model_dir+"/modelparams.txt"
        
        write_model_parameters_Ensb(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                                   min_delta,restore_best_weights,batch_size,epochs,callbacks,
                                   classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                                   training_dataset,train,validation,monitor,mode,metric_name,start_input)
        
        plot_model_history(history,model_dir)
        
        metrics_results=[]
        
        test_tags,tsets_thrshlds_dict=test_tags_thresholds(pdbsets)
        X_testsets=select_testsets(pdbsets,Xy_datasets)
        X_testsets["X_valid"]=[validation,target_val]
        tsets_thrshlds_dict["X_valid"]=0.5
        del train,validation,target_train,target_val, \
        val_inputs,train_inputs,train_decoder_input,valid_decoder_input
        gc.collect()
        test_results_LSTMs2s(model,X_testsets,tsets_thrshlds_dict,start_input,model_dir,
                                 metrics_results,scale_data,scaler,clsf_model)
        
        append_model_parameters_testsets(mparams_file, X_testsets)
        append_thresholds_testsets(mparams_file, tsets_thrshlds_dict)
        
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        
        test_mtr=[i+"_1" for i in test_tags]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')[metric_name].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        del model
        gc.collect()
        if all_data==False and novald==False and int(fld_no)!=-1: break
        break
        # if fold_no==3: break
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print_results_folds(metric_name,mn_mtr_ls)
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/'+tag+'.dir', 'w') as f:
        f.write(results_dir+f'/model_{tag}_fold/fold_{fd_slct}/model.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)

def kfoldLSTMs2sAttention(results_dir,n_folds):
    under_1_1_training, under_1_1_valid, under_lbl, training_dataset = data_info(undersampling,all_data)
    Xy_datasets = load_data(under_lbl)
    X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
    
    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold , auc_per_fold , loss_per_fold = [], [], []
    precision_per_fold, recall_per_fold, aupr_per_fold=[], [], []
    f1_per_fold, specivity_per_fold, mcc_per_fold=[], [], []
    mse_per_fold, mn_mtr_ls=[], []
    scaler = StandardScaler()
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        if all_data==True:
            # Xy_datasets = load_data(under_lbl)
            # X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
            train, validation = X_train[train_index], X_train[validation_index]
            target_train, target_val = y_train[train_index], y_train[validation_index]
        else:
            # Xy_datasets = load_data(under_lbl)
            # X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid
            
        if scale_data==True:
            tests=[validation]+Xy_datasets[4::2] if pdbsets==True else [validation]+Xy_datasets[10::2]
            train, tests, scaler = scale_datasets(train,tests,scaler_type)
            
        print("\nFold ",fold_no)
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        
        tf.keras.backend.clear_session()
        ### PARAMETERS ###
        optimizer_name='adam'
        learning_rate=0.002
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss='binary_crossentropy'
        metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000,name='auprc'),
                 tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000,name='auroc'),
                 'binary_accuracy','mse','Precision','Recall',
                 'TrueNegatives','FalsePositives',
                 'FalseNegatives','TruePositives']
        patience=4
        min_delta=1e-5
        restore_best_weights=True
        validation_split=0.2
        batch_size=1024
        steps_per_epoch=500
        epochs=2
        lstm_units=16
        dense_units=64
        dropout_rate=0.3
        clsf_model="LSTMs2sAtt"
        tag=clsf_model.lower()
        device_type="CPU"
        prog_name=sys.argv[0]
        shuffle=True
        start_input=-2
        monitor,mode,metric_name="val_auroc",'max',"AUROC"
        input_shape=train.shape[1:]
        hidden_input_shape=train.shape[1]
        class_num=np.unique(target_train)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=restore_best_weights,
            mode=mode
        )
        callbacks=[early_stopping]
        classWeight = classWeight_comp(class_num,target_train,sqrtw,classwcomp)
        print(classWeight)
        sample_weight = sampleWeight_comp(classWeight,target_train,sampw)
        ### =========== ###
        
        ### MODEL ###
        model = build_lstms2sAttention_model(input_shape, lstm_units, loss, optimizer, metrics, tag, fold_no)
        
        model.summary()
        
        train_decoder_input=np.full((target_train.shape[0],), start_input)
        valid_decoder_input=np.full((target_val.shape[0],), start_input)

        # Define checkpoint callback
        model_dir=results_dir+'/model_lstms2satt_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=model_dir+'/model_graph_lstms2satt.png')
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
            early_stopping
            # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True, mode='max')
        ]
        val_inputs=[validation,valid_decoder_input]
        train_inputs =[train,train_decoder_input]
        
        device_t="/device:"+device_type_control(model_dir,device_type)+":0"
        
        classWeight = class_wght(classw,classwcomp,classWeight)
        print(classWeight)
        
        validation_data, callbacks = vald_data_callback(all_data,novald,
                                                        val_inputs,target_val,callbacks_check)
        ### FIT MODEL ###
        with tf.device(device_t):
            history = model.fit(
                [train, train_decoder_input], target_train, 
                epochs=epochs, 
                validation_data=validation_data,
                callbacks=callbacks,
                batch_size=batch_size,
                class_weight=classWeight,
                sample_weight=sample_weight,
                shuffle=shuffle
            )
            
        # Save model
        model.save(model_dir+'/model.tf')
        model.save_weights(model_dir+"/weights.tf")

        encoder_model, decoder_model = build_lstms2sAttention_inference_model(model, tag, fold_no)
        
        tf.keras.utils.plot_model(encoder_model, to_file=results_dir+"/model_enc.png", show_shapes=True)
        tf.keras.utils.plot_model(decoder_model, to_file=results_dir+"/model_dec.png", show_shapes=True)
        
        enc_states_val=encoder_model.predict(validation)
        
        probas = decoder_model.predict([valid_decoder_input] + enc_states_val)

        labels = (probas >= 0.5).astype(int)
        y_pred = np.squeeze(np.asarray(labels))
        print(classification_report(target_val, y_pred))
        print(confusion_matrix(target_val,y_pred))
        # scores = decoder_model.evaluate([valid_decoder_input] + validation, target_val, verbose=0)
        # print_validation_results_fold(model, scores, fold_no)
        
        # # print(model.metrics_names)
        
        # specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc = compute_metrics(model,scores)
        
        # loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold, \
        # precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold, \
        # auc_per_fold,aupr_per_fold = \
        # metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
        #                        loss_per_fold,mse_per_fold,recall_per_fold,
        #                        specivity_per_fold,precision_per_fold,acc_per_fold,
        #                        f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold)
        fold_no = fold_no + 1
        
        # print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold)
        
        # print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
        #                                    precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
        #                                    auc_per_fold,aupr_per_fold)
        
        mparams_file=model_dir+"/modelparams.txt"
        
        write_model_parameters_LSTMs2s(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                                   min_delta,restore_best_weights,lstm_units,batch_size,epochs,callbacks,
                                   classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                                   training_dataset,train,validation,start_input,monitor,mode,metric_name)
        
        plot_model_history(history,model_dir)
        
        metrics_results=[]
              
        test_tags,tsets_thrshlds_dict=test_tags_thresholds(pdbsets)
        X_testsets=select_testsets(pdbsets,Xy_datasets)
        test_results_LSTMs2s_encdec(encoder_model, decoder_model,X_testsets,tsets_thrshlds_dict,start_input,model_dir,
                             metrics_results,scale_data,scaler,clsf_model)
        
        append_model_parameters_testsets(mparams_file, X_testsets)
        append_thresholds_testsets(mparams_file, tsets_thrshlds_dict)
        
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        
        test_mtr=[i+"_1" for i in test_tags]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')[metric_name].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        del train,validation,target_train,target_val,model, \
        val_inputs,train_inputs,train_decoder_input,valid_decoder_input
        gc.collect()
        if all_data==False and novald==False: break
        # if fold_no==3: break
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print_results_folds(metric_name,mn_mtr_ls)
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/'+tag+'.dir', 'w') as f:
        f.write(results_dir+f'/model_{tag}_fold/fold_{fd_slct}/model.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)

def kfoldLSTMs2sAttention2(results_dir,n_folds,b_norm=True):
    under_1_1_training, under_1_1_valid, under_lbl, training_dataset = data_info(undersampling,all_data)
    Xy_datasets = load_data(under_lbl)
    X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
    
    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold , auc_per_fold , loss_per_fold = [], [], []
    precision_per_fold, recall_per_fold, aupr_per_fold=[], [], []
    f1_per_fold, specivity_per_fold, mcc_per_fold=[], [], []
    mse_per_fold, mn_mtr_ls=[], []
    scaler = StandardScaler()
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        if all_data==True:
            # Xy_datasets = load_data(under_lbl)
            # X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
            train, validation = X_train[train_index], X_train[validation_index]
            target_train, target_val = y_train[train_index], y_train[validation_index]
        else:
            # Xy_datasets = load_data(under_lbl)
            # X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid
            
        if scale_data==True:
            tests=[validation]+Xy_datasets[4::2] if pdbsets==True else [validation]+Xy_datasets[10::2]
            train, tests, scaler = scale_datasets(train,tests,scaler_type)
            
        print("\nFold ",fold_no)
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        
        tf.keras.backend.clear_session()
        ### PARAMETERS ###
        optimizer_name='adam'
        learning_rate=0.002
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss='binary_crossentropy'
        metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000,name='auprc'),
                 tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000,name='auroc'),
                 'binary_accuracy','mse','Precision','Recall',
                 'TrueNegatives','FalsePositives',
                 'FalseNegatives','TruePositives']
        patience=3
        min_delta=1e-5
        restore_best_weights=True
        validation_split=0.2
        batch_size=512
        steps_per_epoch=500
        epochs=15
        lstm_units=16
        dense_units=64
        dropout_rate=0.3
        clsf_model="LSTMs2sAtt"
        tag=clsf_model.lower()
        device_type="CPU"
        prog_name=sys.argv[0]
        shuffle=True
        start_input=-2
        monitor,mode,metric_name="val_auprc",'max',"AUPRC"
        input_shape=train.shape[1:]
        hidden_input_shape=train.shape[1]
        class_num=np.unique(target_train)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=restore_best_weights,
            mode=mode
        )
        callbacks=[early_stopping]
        classWeight = classWeight_comp(class_num,target_train,sqrtw,classwcomp)
        print(classWeight)
        sample_weight = sampleWeight_comp(classWeight,target_train,sampw)
        ### =========== ###
        
        ### MODEL ###
        if b_norm==True:
            model = build_lstms2sAttention_model_bnorm3(input_shape, lstm_units, loss, optimizer, metrics, tag, fold_no)
        else:
            model = build_lstms2sAttention_model(input_shape, lstm_units, loss, optimizer, metrics, tag, fold_no)
        model.summary()
        
        # train_decoder_input=np.full((target_train.shape[0],), start_input)
        # valid_decoder_input=np.full((target_val.shape[0],), start_input)
        train_decoder_input=np.array(np.random.choice([0, 1], size=(target_train.shape[0])))
        valid_decoder_input=np.array(np.random.choice([0, 1], size=(target_val.shape[0])))
        print(sum(train_decoder_input),sum(valid_decoder_input))
        
        # Define checkpoint callback
        model_dir=results_dir+'/model_'+tag+'_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=model_dir+'/model_graph_'+tag+'.png')
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
            early_stopping
            # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True, mode='max')
        ]
        
        train_inputs =[train,train_decoder_input]
        val_inputs=[validation,valid_decoder_input]
                
        device_t="/device:"+device_type_control(model_dir,device_type)+":0"
        
        classWeight = class_wght(classw,classwcomp,classWeight)
        print(classWeight)
        
        validation_data, callbacks = vald_data_callback(all_data,novald,
                                                        val_inputs,target_val,callbacks_check)
        ### FIT MODEL ###
        with tf.device(device_t):
            history = model.fit(
                train_inputs, target_train, 
                epochs=epochs, 
                validation_data=validation_data,
                callbacks=callbacks,
                batch_size=batch_size,
                class_weight=classWeight,
                sample_weight=sample_weight,
                shuffle=shuffle
            )
            
        # Save model
        model.save(model_dir+'/model.tf')
        model.save_weights(model_dir+"/weights.tf")
        
        probas = model.predict(val_inputs).ravel()
        plot_probas_valid(probas,model_dir)
        labels = (probas >= 0.5).astype(int)
        y_pred = np.squeeze(np.asarray(labels))
        print(classification_report(target_val, y_pred))
        print(confusion_matrix(target_val,y_pred))
        
        scores = model.evaluate(val_inputs, target_val, verbose=0)
        print_validation_results_fold(model, scores, fold_no)
        
        # # print(model.metrics_names)
        
        specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc = compute_metrics(model,scores)
        
        loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold, \
        precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold, \
        auc_per_fold,aupr_per_fold = \
        metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
                                loss_per_fold,mse_per_fold,recall_per_fold,
                                specivity_per_fold,precision_per_fold,acc_per_fold,
                                f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold)
        fold_no = fold_no + 1
        
        print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold)
        
        print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
                                            precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
                                            auc_per_fold,aupr_per_fold)
        
        mparams_file=model_dir+"/modelparams.txt"
        
        write_model_parameters_LSTMs2s(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                                   min_delta,restore_best_weights,lstm_units,batch_size,epochs,callbacks,
                                   classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                                   training_dataset,train,validation,start_input,monitor,mode,metric_name)
        
        plot_model_history(history,model_dir)
        
        metrics_results=[]
        
        test_tags,tsets_thrshlds_dict=test_tags_thresholds(pdbsets)
        X_testsets=select_testsets(pdbsets,Xy_datasets)
        test_results_LSTMs2s(model,X_testsets,tsets_thrshlds_dict,start_input,model_dir,
                             metrics_results,scale_data,scaler,clsf_model)
        
        append_model_parameters_testsets(mparams_file, X_testsets)
        append_thresholds_testsets(mparams_file, tsets_thrshlds_dict)
        
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        
        test_mtr=[i+"_1" for i in test_tags]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')[metric_name].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        del train,validation,target_train,target_val,model, \
        val_inputs,train_inputs,train_decoder_input,valid_decoder_input
        gc.collect()
        if all_data==False and novald==False: break
        # if fold_no==3: break
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print_results_folds(metric_name,mn_mtr_ls)
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/'+tag+'.dir', 'w') as f:
        f.write(results_dir+f'/model_{tag}_fold/fold_{fd_slct}/model.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)

def kfoldGRUs2sAttention(results_dir,n_folds,b_norm=True):
    under_1_1_training, under_1_1_valid, under_lbl, training_dataset = data_info(undersampling,all_data)
    Xy_datasets = load_data(under_lbl)
    X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
    results_dir = create_result_folder(results_dir,X_train.shape[:])
    
    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold , auc_per_fold , loss_per_fold = [], [], []
    precision_per_fold, recall_per_fold, aupr_per_fold=[], [], []
    f1_per_fold, specivity_per_fold, mcc_per_fold=[], [], []
    mse_per_fold, mn_mtr_ls=[], []
    scaler = StandardScaler()
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        if int(fld_no)!=-1 and fold_no!=int(fld_no): 
            fold_no+=1
            continue
        if all_data==True:
            # Xy_datasets = load_data(under_lbl)
            # X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
            train, validation = X_train[train_index], X_train[validation_index]
            target_train, target_val = y_train[train_index], y_train[validation_index]
        else:
            # Xy_datasets = load_data(under_lbl)
            # X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid
            
        if scale_data==True:
            tests=[validation]+Xy_datasets[4::2] if pdbsets==True else [validation]+Xy_datasets[10::2]
            train, tests, scaler = scale_datasets(train,tests,scaler_type)
            
        print("\nFold ",fold_no)
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        
        tf.keras.backend.clear_session()
        ### PARAMETERS ###
        optimizer_name='adam'
        learning_rate=0.002
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss='binary_crossentropy'
        metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000,name='auprc'),
                 tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000,name='auroc'),
                 'binary_accuracy','mse','Precision','Recall',
                 'TrueNegatives','FalsePositives',
                 'FalseNegatives','TruePositives']
        patience=3
        min_delta=1e-5
        restore_best_weights=True
        validation_split=0.2
        batch_size=1024
        steps_per_epoch=500
        epochs=15
        gru_units=32
        dense_units=64
        dropout_rate=0.3
        clsf_model="GRUs2sAtt"
        tag=clsf_model.lower()
        device_type="CPU"
        prog_name=sys.argv[0]
        shuffle=True
        start_input=-2
        monitor,mode,metric_name="val_auprc",'max',"AUPRC"
        input_shape=train.shape[1:]
        class_num=np.unique(target_train)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=restore_best_weights,
            mode=mode
        )
        callbacks=[early_stopping]
        classWeight = classWeight_comp(class_num,target_train,sqrtw,classwcomp)
        print(classWeight)
        sample_weight = sampleWeight_comp(classWeight,target_train,sampw)
        ### =========== ###
        
        ### MODEL ###
        if b_norm==True:
            model = build_grus2sAttention_model_bnorm2(input_shape, gru_units, loss, optimizer, metrics, tag, fold_no)
        else:
            model = build_grus2sAttention_model(input_shape, gru_units, loss, optimizer, metrics, tag, fold_no)
        model.summary()
        
        # train_decoder_input=np.array(np.random.choice([0, 1], size=(target_train.shape[0])))
        # valid_decoder_input=np.array(np.random.choice([0, 1], size=(target_val.shape[0])))
        # print(sum(train_decoder_input),sum(valid_decoder_input))
        train_decoder_input=np.full((target_train.shape[0],), start_input)
        train_decoder_input=np.reshape(train_decoder_input,(train_decoder_input.shape[0],1,1))
        valid_decoder_input=np.full((target_val.shape[0],), start_input)
        valid_decoder_input=np.reshape(valid_decoder_input,(valid_decoder_input.shape[0],1,1))
        
        # Define checkpoint callback
        model_dir=results_dir+'/model_'+tag+'_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=model_dir+'/model_graph_'+tag+'.png')
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
            early_stopping
            # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True, mode='max')
        ]
        
        train_inputs =[train,train_decoder_input]
        val_inputs=[validation,valid_decoder_input]
                
        device_t="/device:"+device_type_control(model_dir,device_type)+":0"
        
        classWeight = class_wght(classw,classwcomp,classWeight)
        print(classWeight)
        
        validation_data, callbacks = vald_data_callback(all_data,novald,
                                                        val_inputs,target_val,callbacks_check)
        ### FIT MODEL ###
        with tf.device(device_t):
            history = model.fit(
                train_inputs, target_train, 
                epochs=epochs, 
                validation_data=validation_data,
                callbacks=callbacks,
                batch_size=batch_size,
                class_weight=classWeight,
                sample_weight=sample_weight,
                shuffle=shuffle
            )
            
        # Save model
        model.save(model_dir+'/model.tf')
        model.save_weights(model_dir+"/weights.tf")
        
        probas = model.predict(val_inputs).ravel()
        plot_probas_valid(probas,model_dir)
        plot_distrb_pred_valid(probas,target_val,model_dir)
        labels = (probas >= 0.5).astype(int)
        y_pred = np.squeeze(np.asarray(labels))
        print(classification_report(target_val, y_pred))
        print(confusion_matrix(target_val,y_pred))
        
        scores = model.evaluate(val_inputs, target_val, verbose=0)
        print_validation_results_fold(model, scores, fold_no)
        
        # # print(model.metrics_names)
        
        specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc = compute_metrics(model,scores)
        
        loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold, \
        precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold, \
        auc_per_fold,aupr_per_fold = \
        metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
                                loss_per_fold,mse_per_fold,recall_per_fold,
                                specivity_per_fold,precision_per_fold,acc_per_fold,
                                f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold)
        fold_no = fold_no + 1
        
        print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold)
        
        print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
                                            precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
                                            auc_per_fold,aupr_per_fold)
        
        mparams_file=model_dir+"/modelparams.txt"
        
        write_model_parameters_GRUs2s(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                                   min_delta,restore_best_weights,gru_units,batch_size,epochs,callbacks,
                                   classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                                   training_dataset,train,validation,start_input,monitor,mode,metric_name)
        
        plot_model_history(history,model_dir)
        del train,validation,target_train,target_val, \
        val_inputs,train_inputs,train_decoder_input,valid_decoder_input
        gc.collect()
        metrics_results=[]
        
        test_tags,tsets_thrshlds_dict=test_tags_thresholds(pdbsets)
        X_testsets=select_testsets(pdbsets,Xy_datasets)
        test_results_LSTMs2s(model,X_testsets,tsets_thrshlds_dict,start_input,model_dir,
                             metrics_results,scale_data,scaler,clsf_model)
        
        append_model_parameters_testsets(mparams_file, X_testsets)
        append_thresholds_testsets(mparams_file, tsets_thrshlds_dict)
        
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        
        test_mtr=[i+"_1" for i in test_tags]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')[metric_name].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        del model
        gc.collect()
        if all_data==False and novald==False and int(fld_no)!=-1: break
        # if fold_no==3: break
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print_results_folds(metric_name,mn_mtr_ls)
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/'+tag+'.dir', 'w') as f:
        f.write(results_dir+f'/model_{tag}_fold/fold_{fd_slct}/model.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)


def kfoldLSTMs2s(results_dir,n_folds):
    under_1_1_training, under_1_1_valid, under_lbl, training_dataset = data_info(undersampling,all_data)
    Xy_datasets = load_data(under_lbl)
    X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
    
    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold , auc_per_fold , loss_per_fold = [], [], []
    precision_per_fold, recall_per_fold, aupr_per_fold=[], [], []
    f1_per_fold, specivity_per_fold, mcc_per_fold=[], [], []
    mse_per_fold, mn_mtr_ls=[], []
    scaler = StandardScaler()
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        if all_data==True:
            train, validation = X_train[train_index], X_train[validation_index]
            target_train, target_val = y_train[train_index], y_train[validation_index]
        else:
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid

        if scale_data==True:
            tests=[validation]+Xy_datasets[4::2] if pdbsets==True else [validation]+Xy_datasets[10::2]
            train, tests, scaler = scale_datasets(train,tests,scaler_type)

        print("\nFold ",fold_no)
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        
        tf.keras.backend.clear_session()
        ### PARAMETERS ###
        optimizer_name='adam'
        learning_rate=0.002
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss='binary_crossentropy'
        metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000,name='auprc'),
                 tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000,name='auroc'),
                 'binary_accuracy','mse','Precision','Recall',
                 'TrueNegatives','FalsePositives',
                 'FalseNegatives','TruePositives']
        patience=4
        min_delta=1e-5
        restore_best_weights=True
        validation_split=0.2
        batch_size=1024
        steps_per_epoch=500
        epochs=3
        lstm_units=16
        dense_units=64
        dropout_rate=0.3
        clsf_model="LSTMs2s"
        tag=clsf_model.lower()
        device_type="CPU"
        prog_name=sys.argv[0]
        shuffle=True
        start_input=-2
        monitor,mode,metric_name="val_auroc",'max',"AUROC"
        input_shape=train.shape[1:]
        hidden_input_shape=train.shape[1]
        class_num=np.unique(target_train)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=restore_best_weights,
            mode=mode
        )
        callbacks=[early_stopping]
        classWeight = classWeight_comp(class_num,target_train,sqrtw,classwcomp)
        print(classWeight)
        sample_weight = sampleWeight_comp(classWeight,target_train,sampw)
        ### =========== ###
        
        ### MODEL ###
        model = build_lstms2s_model(input_shape, lstm_units, loss, 
                            optimizer, metrics, tag, fold_no)
        
        model.summary()
        
        train_decoder_input=np.full((target_train.shape[0],), start_input)
        valid_decoder_input=np.full((target_val.shape[0],), start_input)

        # Define checkpoint callback
        model_dir=results_dir+'/model_lstms2s_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=model_dir+'/model_graph_lstms2s.png')
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
            early_stopping
            # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True, mode='max')
        ]
        
        val_inputs=[validation,valid_decoder_input]
        train_inputs =[train,train_decoder_input]
        
        device_t="/device:"+device_type_control(model_dir,device_type)+":0"
        
        classWeight = class_wght(classw,classwcomp,classWeight)
        print(classWeight)
        
        validation_data, callbacks = vald_data_callback(all_data,novald,
                                                        val_inputs,target_val,callbacks_check)
        ### FIT MODEL ###
        with tf.device(device_t):
            history = model.fit(
                train_inputs, target_train, 
                epochs=epochs, 
                validation_data=validation_data,
                callbacks=callbacks,
                batch_size=batch_size,
                class_weight=classWeight,
                sample_weight=sample_weight,
                shuffle=shuffle
            )
            
        # Save model
        model.save(model_dir+'/model.tf')
        model.save_weights(model_dir+"/weights.tf")
                
        encoder_model, decoder_model = build_lstms2s_inference_model(model, tag, fold_no)
        
        tf.keras.utils.plot_model(encoder_model, to_file=results_dir+"/model_enc.png", show_shapes=True)
        tf.keras.utils.plot_model(decoder_model, to_file=results_dir+"/model_dec.png", show_shapes=True)
        
        enc_states_val=encoder_model.predict(validation)

        probas = decoder_model.predict([valid_decoder_input] + enc_states_val)

        labels = (probas >= 0.5).astype(int)
        y_pred = np.squeeze(np.asarray(labels))
        print(classification_report(target_val, y_pred))
        print(confusion_matrix(target_val,y_pred))
        # scores = decoder_model.evaluate([valid_decoder_input] + validation, target_val, verbose=0)
        # print_validation_results_fold(model, scores, fold_no)
        
        # # print(model.metrics_names)
        
        # specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc = compute_metrics(model,scores)
        
        # loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold, \
        # precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold, \
        # auc_per_fold,aupr_per_fold = \
        # metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
        #                        loss_per_fold,mse_per_fold,recall_per_fold,
        #                        specivity_per_fold,precision_per_fold,acc_per_fold,
        #                        f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold)
        fold_no = fold_no + 1
        
        # print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold)
        
        # print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
        #                                    precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
        #                                    auc_per_fold,aupr_per_fold)
        
        mparams_file=model_dir+"/modelparams.txt"
        
        write_model_parameters_LSTMs2s(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                                   min_delta,restore_best_weights,lstm_units,batch_size,epochs,callbacks,
                                   classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                                   training_dataset,train,validation,start_input,monitor,mode,metric_name)
        
        plot_model_history(history,model_dir)
        
        metrics_results=[]
        
        test_tags,tsets_thrshlds_dict=test_tags_thresholds(pdbsets)
        X_testsets=select_testsets(pdbsets,Xy_datasets)
        test_results_LSTMs2s_encdec(encoder_model, decoder_model,X_testsets,tsets_thrshlds_dict,start_input,model_dir,
                             metrics_results,scale_data,scaler,clsf_model)
        
        append_model_parameters_testsets(mparams_file, X_testsets)
        append_thresholds_testsets(mparams_file, tsets_thrshlds_dict)
        
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        
        test_mtr=[i+"_1" for i in test_tags]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')[metric_name].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        del train,validation,target_train,target_val,model, \
        val_inputs,train_inputs,train_decoder_input,valid_decoder_input
        gc.collect()
        if all_data==False and novald==False: break
        # if fold_no==3: break
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print_results_folds(metric_name,mn_mtr_ls)
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/'+tag+'.dir', 'w') as f:
        f.write(results_dir+f'/model_{tag}_fold/fold_{fd_slct}/model.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)  

def kfoldLSTMs2s2(results_dir,n_folds,b_norm=True):
    under_1_1_training, under_1_1_valid, under_lbl, training_dataset = data_info(undersampling,all_data)
    Xy_datasets = load_data(under_lbl)
    X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold , auc_per_fold , loss_per_fold = [], [], []
    precision_per_fold, recall_per_fold, aupr_per_fold=[], [], []
    f1_per_fold, specivity_per_fold, mcc_per_fold=[], [], []
    mse_per_fold, mn_mtr_ls=[], []
    scaler = StandardScaler()
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        if all_data==True:
            train, validation = X_train[train_index], X_train[validation_index]
            target_train, target_val = y_train[train_index], y_train[validation_index]
        else:
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid
            
        if scale_data==True:
            tests=[validation]+Xy_datasets[4::2] if pdbsets==True else [validation]+Xy_datasets[10::2]
            train, tests, scaler = scale_datasets(train,tests,scaler_type)
        # print(train[0,:,:])
        # break
        print("\nFold ",fold_no)
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        
        tf.keras.backend.clear_session()
        ### PARAMETERS ###
        optimizer_name='adam'
        learning_rate=0.002
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss='binary_crossentropy'
        metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000,name='auprc'),
                 tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000,name='auroc'),
                 'binary_accuracy','mse','Precision','Recall',
                 'TrueNegatives','FalsePositives',
                 'FalseNegatives','TruePositives']
        patience=3
        min_delta=1e-5
        restore_best_weights=True
        validation_split=0.2
        batch_size=1024
        steps_per_epoch=500
        epochs=15
        lstm_units=32
        dense_units=64
        dropout_rate=0.5
        clsf_model="LSTMs2s"
        tag=clsf_model.lower()
        device_type="CPU"
        prog_name=sys.argv[0]
        shuffle=True
        start_input=-2
        monitor,mode,metric_name="val_auprc",'max',"AUPRC"
        input_shape=train.shape[1:]
        hidden_input_shape=train.shape[1]
        class_num=np.unique(target_train)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=restore_best_weights,
            mode=mode
        )
        callbacks=[early_stopping]
        classWeight = classWeight_comp(class_num,target_train,sqrtw,classwcomp)
        print(classWeight)
        sample_weight = sampleWeight_comp(classWeight,target_train,sampw)
        ### =========== ###
        
        ### MODEL ###
        if b_norm==True:
            model = build_lstms2s_model_bnorm2(input_shape, lstm_units, loss, 
                                optimizer, metrics, tag, fold_no)
        else:
            model = build_lstms2s_model(input_shape, lstm_units, loss, 
                                optimizer, metrics, tag, fold_no)
        
        model.summary()

        train_decoder_input=np.full((target_train.shape[0],), start_input)
        valid_decoder_input=np.full((target_val.shape[0],), start_input)
        # train_decoder_input=np.array(np.random.choice([0, 1], size=(target_train.shape[0]),p=[0.9,0.1]))
        # valid_decoder_input=np.array(np.random.choice([0, 1], size=(target_val.shape[0]),p=[0.9,0.1]))
        # print(sum(train_decoder_input),sum(train_decoder_input)/len(train_decoder_input)*100,
        #       sum(valid_decoder_input),sum(valid_decoder_input)/len(valid_decoder_input)*100)
        # train_decoder_input=np.random.uniform(size=target_train.shape[0])
        # valid_decoder_input=np.random.uniform(size=target_val.shape[0])
        # count, bins, ignored = plt.hist(train_decoder_input, 15, density=True)
        # plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        # plt.show()
        # plt.close()
        # Define checkpoint callback
        model_dir=results_dir+'/model_'+tag+'_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=model_dir+'/model_graph_'+tag+'.png')
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
            early_stopping
            # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True, mode='max')
        ]
        
        train_inputs =[train,train_decoder_input]
        val_inputs=[validation,valid_decoder_input]
        device_t="/device:"+device_type_control(model_dir,device_type)+":0"
        
        classWeight = class_wght(classw,classwcomp,classWeight)
        print(classWeight)
        
        validation_data, callbacks = vald_data_callback(all_data,novald,
                                                        val_inputs,target_val,callbacks_check)
        ### FIT MODEL ###
        with tf.device(device_t):
            history = model.fit(
                train_inputs, target_train, 
                epochs=epochs, 
                validation_data=validation_data,
                callbacks=callbacks,
                batch_size=batch_size,
                class_weight=classWeight,
                sample_weight=sample_weight,
                shuffle=shuffle
            )
            
        # Save model
        model.save(model_dir+'/model.tf')
        model.save_weights(model_dir+"/weights.tf")
        
        probas = model.predict(val_inputs).ravel()
        plot_probas_valid(probas,model_dir)
        plot_distrb_pred_valid(probas,target_val,model_dir)
        labels = (probas >= 0.5).astype(int)
        y_pred = np.squeeze(np.asarray(labels))
        print(classification_report(target_val, y_pred))
        print(confusion_matrix(target_val,y_pred))
        
        scores = model.evaluate(val_inputs, target_val, verbose=0)
        print_validation_results_fold(model, scores, fold_no)
        
        # # print(model.metrics_names)
        
        specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc = compute_metrics(model,scores)
        
        loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold, \
        precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold, \
        auc_per_fold,aupr_per_fold = \
        metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
                                loss_per_fold,mse_per_fold,recall_per_fold,
                                specivity_per_fold,precision_per_fold,acc_per_fold,
                                f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold)
        fold_no = fold_no + 1
        
        print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold)
        
        print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
                                            precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
                                            auc_per_fold,aupr_per_fold)
        
        mparams_file=model_dir+"/modelparams.txt"
        
        write_model_parameters_LSTMs2s(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                                   min_delta,restore_best_weights,lstm_units,batch_size,epochs,callbacks,
                                   classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                                   training_dataset,train,validation,start_input,monitor,mode,metric_name)
        
        plot_model_history(history,model_dir)
        
        metrics_results=[]
        
        test_tags,tsets_thrshlds_dict=test_tags_thresholds(pdbsets)
        X_testsets=select_testsets(pdbsets,Xy_datasets)
        test_results_LSTMs2s(model,X_testsets,tsets_thrshlds_dict,start_input,model_dir,
                             metrics_results,scale_data,scaler,clsf_model)
        
        append_model_parameters_testsets(mparams_file, X_testsets)
        append_thresholds_testsets(mparams_file, tsets_thrshlds_dict)
        
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        
        test_mtr=[i+"_1" for i in test_tags]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')[metric_name].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        del train,validation,target_train,target_val,model, \
        val_inputs,train_inputs,train_decoder_input,valid_decoder_input
        gc.collect()
        if all_data==False and novald==False: break
        # if fold_no==3: break
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print_results_folds(metric_name,mn_mtr_ls)
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/'+tag+'.dir', 'w') as f:
        f.write(results_dir+f'/model_{tag}_fold/fold_{fd_slct}/model.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)
    

def kfoldGRUs2s(results_dir,n_folds,b_norm=True):
    under_1_1_training, under_1_1_valid, under_lbl, training_dataset = data_info(undersampling,all_data)
    Xy_datasets = load_data(under_lbl)
    X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
    results_dir = create_result_folder(results_dir,X_train.shape[:])
    
    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold , auc_per_fold , loss_per_fold = [], [], []
    precision_per_fold, recall_per_fold, aupr_per_fold=[], [], []
    f1_per_fold, specivity_per_fold, mcc_per_fold=[], [], []
    mse_per_fold, mn_mtr_ls=[], []
    scaler = StandardScaler()
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        if int(fld_no)!=-1 and fold_no!=int(fld_no): 
            fold_no+=1
            continue
        if all_data==True:
            train, validation = X_train[train_index], X_train[validation_index]
            target_train, target_val = y_train[train_index], y_train[validation_index]
        else:
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid
            
        if scale_data==True:
            tests=[validation]+Xy_datasets[4::2] if pdbsets==True else [validation]+Xy_datasets[10::2]
            train, tests, scaler = scale_datasets(train,tests,scaler_type)
        # print(train[0,:,:])
        # break
        print("\nFold ",fold_no)
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        
        tf.keras.backend.clear_session()
        ### PARAMETERS ###
        optimizer_name='adam'
        learning_rate=0.002
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss='binary_crossentropy'
        metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000,name='auprc'),
                 tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000,name='auroc'),
                 'binary_accuracy','mse','Precision','Recall',
                 'TrueNegatives','FalsePositives',
                 'FalseNegatives','TruePositives']
        patience=3
        min_delta=1e-5
        restore_best_weights=True
        validation_split=0.2
        batch_size=1024
        steps_per_epoch=500
        epochs=15
        lstm_units=16
        gru_units=32
        dense_units=64
        dropout_rate=0.5
        clsf_model="GRUs2s"
        tag=clsf_model.lower()
        device_type="CPU"
        prog_name=sys.argv[0]
        shuffle=True
        start_input=-2
        monitor,mode,metric_name="val_auprc",'max',"AUPRC"
        input_shape=train.shape[1:]
        hidden_input_shape=train.shape[1]
        class_num=np.unique(target_train)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=restore_best_weights,
            mode=mode
        )
        callbacks=[early_stopping]
        classWeight = classWeight_comp(class_num,target_train,sqrtw,classwcomp)
        print(classWeight)
        sample_weight = sampleWeight_comp(classWeight,target_train,sampw)
        ### =========== ###
        # target_train=np.reshape(target_train, (target_train.shape[0],1))
        # target_val=np.reshape(target_val, (target_val.shape[0],1))
        # for i in range(5,len(Xy_datasets),2):
        #     Xy_datasets[i]=np.reshape(Xy_datasets[i], (Xy_datasets[i].shape[0],1))
        ### MODEL ###
        if b_norm==True:
            model = build_grus2s_model_bnorm(input_shape, gru_units, 
                                             loss, optimizer, metrics, tag, fold_no)
        else:
            model = build_grus2s_model(input_shape, gru_units, 
                                       loss, optimizer, metrics, tag, fold_no)
        
        model.summary()

        train_decoder_input=np.full((target_train.shape[0],), start_input)
        train_decoder_input=np.reshape(train_decoder_input,(train_decoder_input.shape[0],1,1))
        valid_decoder_input=np.full((target_val.shape[0],), start_input)
        valid_decoder_input=np.reshape(valid_decoder_input,(valid_decoder_input.shape[0],1,1))

        # Define checkpoint callback
        model_dir=results_dir+'/model_'+tag+'_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=model_dir+'/model_graph_'+tag+'.png')
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
            early_stopping
            # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True, mode='max')
        ]
        
        train_inputs =[train,train_decoder_input]
        val_inputs=[validation,valid_decoder_input]
        device_t="/device:"+device_type_control(model_dir,device_type)+":0"
        
        classWeight = class_wght(classw,classwcomp,classWeight)
        print(classWeight)
        
        validation_data, callbacks = vald_data_callback(all_data,novald,
                                                        val_inputs,target_val,callbacks_check)
        ### FIT MODEL ###
        with tf.device(device_t):
            history = model.fit(
                train_inputs, target_train, 
                epochs=epochs, 
                validation_data=validation_data,
                callbacks=callbacks,
                batch_size=batch_size,
                class_weight=classWeight,
                sample_weight=sample_weight,
                shuffle=shuffle
            )
            
        # Save model
        model.save(model_dir+'/model.tf')
        model.save_weights(model_dir+"/weights.tf")
        
        probas = model.predict(val_inputs).ravel()
        plot_probas_valid(probas,model_dir)
        plot_distrb_pred_valid(probas,target_val,model_dir)
        labels = (probas >= 0.5).astype(int)
        y_pred = np.squeeze(np.asarray(labels))
        print(classification_report(target_val, y_pred))
        print(confusion_matrix(target_val,y_pred))
        
        scores = model.evaluate(val_inputs, target_val, verbose=0)
        print_validation_results_fold(model, scores, fold_no)
        
        # # print(model.metrics_names)
        
        specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc = compute_metrics(model,scores)
        
        loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold, \
        precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold, \
        auc_per_fold,aupr_per_fold = \
        metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
                                loss_per_fold,mse_per_fold,recall_per_fold,
                                specivity_per_fold,precision_per_fold,acc_per_fold,
                                f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold)
        fold_no = fold_no + 1
        
        print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold)
        
        print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
                                            precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
                                            auc_per_fold,aupr_per_fold)
        
        mparams_file=model_dir+"/modelparams.txt"
        
        write_model_parameters_GRUs2s(mparams_file,model,prog_name,device_t,optimizer_name,loss,metrics,patience,
                                   min_delta,restore_best_weights,gru_units,batch_size,epochs,callbacks,
                                   classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                                   training_dataset,train,validation,start_input,monitor,mode,metric_name)
        
        plot_model_history(history,model_dir)
        del train,validation,target_train,target_val, \
        val_inputs,train_inputs,train_decoder_input,valid_decoder_input
        gc.collect()
        metrics_results=[]
        
        test_tags,tsets_thrshlds_dict=test_tags_thresholds(pdbsets)
        X_testsets=select_testsets(pdbsets,Xy_datasets)
        test_results_LSTMs2s(model,X_testsets,tsets_thrshlds_dict,start_input,model_dir,
                             metrics_results,scale_data,scaler,clsf_model)
        
        append_model_parameters_testsets(mparams_file, X_testsets)
        append_thresholds_testsets(mparams_file, tsets_thrshlds_dict)
        
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        
        test_mtr=[i+"_1" for i in test_tags]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')[metric_name].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        del model
        gc.collect()
        if all_data==False and novald==False and int(fld_no)!=-1: break
        # if fold_no==3: break
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print_results_folds(metric_name,mn_mtr_ls)
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/'+tag+'.dir', 'w') as f:
        f.write(results_dir+f'/model_{tag}_fold/fold_{fd_slct}/model.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)


def AttentionLSTM(results_dir):
    if undersampling==True:
        under_1_1_training=True
        under_1_1_valid=True
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl="_down"
    else:
        under_1_1_training=False
        under_1_1_valid=False
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl=""
    training_dataset="df_alldataW_all.csv"
    X_train, y_train, X_valid, y_valid, \
    X_355, y_355, \
    X_448, y_448 = load_data(under_lbl)
    
    tf.keras.backend.clear_session()
    optimizer_name='adam'
    learning_rate=0.002
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss='binary_crossentropy'
    metrics=[tf.keras.metrics.AUC(curve='PR')]
    patience=4
    min_delta=0.001
    restore_best_weights=True
    validation_split=0.2
    batch_size=1024
    steps_per_epoch=500
    epochs=5
    padding='same'
    strides=(1, 1) 
    kernel_size=5
    n_filter=48
    max_pool=(2,2)
    gru_unit=8
    TIME_STEPS = 2
    INPUT_DIM = 18
    lstm_units = 8
    nb_classes = 1
    num_layr=60
    input_shape=X_train.shape[1:]
    early_stopping = keras.callbacks.EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights
    )
    callbacks=[early_stopping]
    w0=X_train.shape[0]/(2*(X_train.shape[0]-np.sum(y_train,dtype=int)))
    w1=X_train.shape[0]/(2*np.sum(y_train,dtype=int))
    class_weight={0: w0, 1: w1}
    #class_weight={0: 0.55, 1: 4.97}
    if sqrtw==True:
    	w0=math.sqrt(w0)
    	w1=math.sqrt(w1)
    if classwcomp==True: 
        class_weight={0: w0, 1: w1}
    else:
        class_weight={0: 0.55, 1: 4.97}
    print(class_weight)
    if sampw==True:
        sample_weight=np.array([w0 if i==0 else w1 for i in y_train.tolist()])
    else:
        sample_weight=None
    shuffle=True
    print(X_train.shape[0],X_train.shape[1],X_train.shape[2])
    
    start_time = time.time()
    start = time.process_time()
    
    inputs = tf.keras.Input(shape=X_train.shape[1:])
    # layer=layers.Bidirectional(layers.LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), dropout=0.3,return_sequences=True), name='bilstm')(inputs)
    # layer1=layers.Bidirectional(layers.LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), dropout=0.3,return_sequences=True), name='bilstm1')(layer)
    # layer2=layers.Bidirectional(layers.LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), dropout=0.3,return_sequences=True), name='bilstm2')(layer1)
    # layer2 = concatenate([layer, layer2])  
    
    # layer3=layers.Bidirectional(layers.LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), dropout=0.3,return_sequences=True), name='bilstm3')(layer2)
    # layer4=layers.Bidirectional(layers.LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), dropout=0.3,return_sequences=True), name='bilstm4')(layer3)
    # layer4 = concatenate([layer2, layer4]) 
    
    # layer5=layers.Bidirectional(layers.LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), dropout=0.3,return_sequences=True), name='bilstm3')(layer4)
    # layer6=layers.Bidirectional(layers.LSTM(lstm_units,activation='softsign',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), dropout=0.3,return_sequences=True), name='bilstm4')(layer5)
    # layer5 = concatenate([layer4, layer6]) 
    layr=[layers.Bidirectional(layers.LSTM(lstm_units,activation='tanh',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), dropout=0.6,return_sequences=True), name='bilstm')(inputs)]
    for i in range(0,num_layr+2,2):
        layr.append(layers.Bidirectional(layers.LSTM(lstm_units,activation='tanh',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), dropout=0.6,return_sequences=True), name='bilstm'+str(i+1))(layr[i]))
        layr.append(layers.Bidirectional(layers.LSTM(lstm_units,activation='tanh',recurrent_activation='sigmoid',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), dropout=0.6,return_sequences=True), name='bilstm'+str(i+2))(layr[i+1]))
        layr[i+2]=concatenate([layr[i], layr[i+2]])
    # layer6=layers.Dense(96, activation='relu')(layer4)
    # layer7=layers.Dropout(0.3)(layer6)
    # layer8=layers.Dense(1,activation='sigmoid')(layer7)
    
    # attention = layers.Dense(1, activation='tanh')(layer4)
    # attention = layers.Flatten()(attention)
    # attention = layers.Activation('softmax')(attention)
    # attention = layers.RepeatVector(16)(attention)
    # attention = layers.Permute([2, 1], name='attention_vec')(attention)
    # attention_mul = add([layer, attention])
    # out_attention_mul = Flatten()(attention_mul)
    # output = Dense(2, activation='sigmoid')(out_attention_mul)
    # model = Model(inputs=inputs, outputs=output)
    
    attention_mul = attention_3d_block(layr[num_layr])
    attention_flatten = Flatten()(attention_mul)
    drop2 = Dropout(0.15)(attention_flatten)
    output = Dense(1, activation='sigmoid')(drop2)
    model = Model(inputs=inputs, outputs=output)
    # model = Model(inputs=inputs, outputs=layer8)
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    if tf.test.is_gpu_available():
        print("GPUs available: ")
        print(tf.config.experimental.list_physical_devices('XLA_GPU'))
        print(tf.config.experimental.list_physical_devices('GPU'))
    print("Devices List: ")
    print(device_lib.list_local_devices())
    if device_type=="GPU":
        device_t="/device:GPU:0"
    else:
        device_t="/device:CPU:0"
        
    if classw==True or classwcomp==True:
        class_weight=class_weight
    else: 
        class_weight=None
    with tf.device(device_t):
        history = model.fit(
            x=X_train, y=y_train, 
            epochs=epochs, 
            validation_data=(X_valid,y_valid),
            callbacks=callbacks,
            batch_size=batch_size,
            class_weight=class_weight,
            sample_weight=sample_weight,
            shuffle=shuffle
        )
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    
    print("process time: ", ptime, "real time: ", rtime)
    
    # timestr = time.strftime("%Y%m%d_%H%M%S")
    # print("timestr: ", timestr)
    # results_dir="./results"
    # if not os.path.exists(results_dir): os.mkdir(results_dir)
    # timestr='20210508_150602'
    # model = keras.models.load_model(results_dir+'/'+'deeplearning3tW_d1_model_20210508_150602')
    # history_df=pd.read_csv(model_dir+'/'+'deeplearning3tW_d1_model_20210508_150602_history.csv', index_col=0, header=0)
    model_dir=results_dir+'/model_ensb'
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model.save(model_dir+"/model.h5")
    model.save_weights(model_dir+"/weights.h5")
    orig_stdout = sys.stdout
    f = open(model_dir+"/devices.txt", 'w+')
    sys.stdout = f
    for device in device_lib.list_local_devices():
        print(device)
    sys.stdout = orig_stdout
    f.close()
    mparams_file=model_dir+"/modelparams.txt"
    with open(mparams_file, 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.write("program name: "+sys.argv[0]+ '\n')
        f.write("device used: "+device_t+ '\n')
        f.write("optimizer: "+optimizer_name+ '\n')
        f.write("loss: "+loss+ '\n')
        f.write("metrics: "+str(metrics)+ '\n')
        f.write("patience: "+ str(patience)+ '\n')
        f.write("min_delta: "+str(min_delta)+ '\n')
        f.write("restore_best_weights: "+str(restore_best_weights)+ '\n')
        f.write("gru_unit: "+str(gru_unit)+ '\n')
        # f.write("validation_split: "+str(validation_split)+ '\n')
        f.write("batch_size: "+str(batch_size)+ '\n')
        # f.write("steps_per_epoch: "+str(steps_per_epoch)+ '\n')
        f.write("epochs: "+str(epochs)+ '\n')
        # f.write("padding: "+padding+ '\n')
        # f.write("strides: "+str(strides)+ '\n')
        # f.write("kernel_size: "+str(kernel_size)+ '\n')
        # f.write("n_filter: "+str(n_filter)+ '\n')
        # f.write("max_pool: "+str(max_pool)+ '\n')
        f.write("callbacks: "+"[early_stopping]"+ '\n')
        f.write("class_weight: "+str(class_weight)+ '\n')
        f.write("shuffle: "+str(shuffle)+ '\n')
        f.write("Arguments: "+str(vars(args))+ '\n')
        f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ '\n')
        f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ '\n')
        f.write("Test dataset 355 undersampled (1:1): "+str(under_1_1_test_355)+ '\n')
        f.write("Test dataset 448 undersampled (1:1): "+str(under_1_1_test_448)+ '\n')
        f.write("Training dataset: "+training_dataset+ '\n')
        # f.write("The length of training dataset: "+str(len(df1))+ '\n')
        # f.write("The length of case study(cs) dataset: "+str(len(df_cs))+ '\n')
        # f.write("The length of training dataset w/o cs: "+str(len(df))+ '\n')
        f.write("The number of columns of training dataset w/o label: "+str(len(X_train[0][0]))+ '\n')
        f.write("The size of sliding windows : "+str(len(X_train[0]))+ '\n')
        f.write("The length of training dataset after splitting: "+str(len(X_train))+ '\n')
        f.write("The length of validation dataset: "+str(len(X_valid))+ '\n')
        f.close()
        
    plot_model_history(history,model_dir)
    
    metrics_results=[]
    
    print("Dtestset 355 \n")
    test_dataset_355="df_alldataW_Dset_355.csv"
    testset_prediction_eval(model,X_355,y_355,[],model_dir,metrics_results,"X_355")
    
    print("Dtestset 448 \n")
    test_dataset_448="df_alldataW_Dset_448.csv"
    testset_prediction_eval(model,X_448,y_448,[],model_dir,metrics_results,"X_448")
    
    with open(mparams_file, 'a') as f:
        f.write("Test dataset: "+test_dataset_355+ '\n')
        f.write("The length of test dataset 355: "+str(len(X_355))+ '\n')
        f.write("Test dataset: "+test_dataset_448+ '\n')
        f.write("The length of test dataset 448: "+str(len(X_448))+ '\n')
        f.close()
    df_metrics_all=pd.concat(metrics_results)
    print(df_metrics_all)
    
    df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
    df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")

def AttentionGRU(results_dir):
    if undersampling==True:
        under_1_1_training=True
        under_1_1_valid=True
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl="_down"
    else:
        under_1_1_training=False
        under_1_1_valid=False
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl=""
    training_dataset="df_alldataW_all.csv"
    X_train, y_train, X_valid, y_valid, \
    X_355, y_355, \
    X_448, y_448 = load_data(under_lbl)

    tf.keras.backend.clear_session()
    optimizer_name='adam'
    learning_rate=0.002
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss='binary_crossentropy'
    metrics=[tf.keras.metrics.AUC(curve='PR')]
    patience=4
    min_delta=0.001
    restore_best_weights=True
    validation_split=0.2
    batch_size=512
    steps_per_epoch=500
    epochs=3
    padding='same'
    strides=(1, 1) 
    kernel_size=5
    n_filter=48
    max_pool=(2,2)
    gru_unit=8
    TIME_STEPS = 2
    INPUT_DIM = 18
    lstm_units = 8
    nb_classes = 1
    num_layr=6 # give even number. if odd, it will be rounded next even number.
    input_shape=X_train.shape[1:]
    early_stopping = keras.callbacks.EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights
    )
    callbacks=[early_stopping]
    w0=X_train.shape[0]/(2*(X_train.shape[0]-np.sum(y_train,dtype=int)))
    w1=X_train.shape[0]/(2*np.sum(y_train,dtype=int))
    class_weight={0: w0, 1: w1}
    #class_weight={0: 0.55, 1: 4.97}
    if sqrtw==True:
    	w0=math.sqrt(w0)
    	w1=math.sqrt(w1)
    if classwcomp==True: 
        class_weight={0: w0, 1: w1}
    else:
        class_weight={0: 0.55, 1: 4.97}
    print(class_weight)
    if sampw==True:
        sample_weight=np.array([w0 if i==0 else w1 for i in y_train.tolist()])
    else:
        sample_weight=None
    shuffle=True
    print(X_train.shape[0],X_train.shape[1],X_train.shape[2])
    
    start_time = time.time()
    start = time.process_time()
    
    inputs = tf.keras.Input(shape=X_train.shape[1:])
    # layer=layers.Bidirectional(layers.GRU(gru_unit,return_sequences=True), name='bigru')(inputs)
    # layer1=layers.Bidirectional(layers.GRU(gru_unit,return_sequences=True), name='bigru1')(layer)
    # layer2=layers.Bidirectional(layers.GRU(gru_unit,return_sequences=True), name='bigru2')(layer1)
    # layer2 = concatenate([layer, layer2])  
    
    # layer3=layers.Bidirectional(layers.GRU(gru_unit,return_sequences=True), name='bigru3')(layer2)
    # layer4=layers.Bidirectional(layers.GRU(gru_unit,return_sequences=True), name='bigru4')(layer3)
    # layer4 = concatenate([layer2, layer4]) 
    
    # layer5=layers.Bidirectional(layers.GRU(gru_unit,return_sequences=True), name='bigru5')(layer4)
    # layer6=layers.Bidirectional(layers.GRU(gru_unit,return_sequences=True), name='bigru6')(layer5)
    # layer6 = concatenate([layer4, layer6]) 
    layr=[layers.Bidirectional(layers.GRU(gru_unit,return_sequences=True), name='bigru')(inputs)]
    for i in range(0,num_layr,2):
        layr.append(layers.Bidirectional(layers.GRU(gru_unit,return_sequences=True), name='bigru'+str(i+1))(layr[i]))
        layr.append(layers.Bidirectional(layers.GRU(gru_unit,return_sequences=True), name='bigru'+str(i+2))(layr[i+1]))
        layr[i+2]=concatenate([layr[i], layr[i+2]])
    # layr=[layers.Bidirectional(layers.LSTM(lstm_units,activation='tanh',return_sequences=True), name='bilstm')(inputs)]
    # for i in range(0,num_layr+2,2):
    #     layr.append(layers.Bidirectional(layers.LSTM(lstm_units,activation='tanh',return_sequences=True), name='bilstm'+str(i+1))(layr[i]))
    #     layr.append(layers.Bidirectional(layers.LSTM(lstm_units,activation='tanh',return_sequences=True), name='bilstm'+str(i+2))(layr[i+1]))
    #     layr[i+2]=concatenate([layr[i], layr[i+2]])
    # layer6=layers.Dense(96, activation='relu')(layer4)
    # layer7=layers.Dropout(0.3)(layer6)
    # layer8=layers.Dense(1,activation='sigmoid')(layer7)
    
    # attention = layers.Dense(1, activation='tanh')(layer4)
    # attention = layers.Flatten()(attention)
    # attention = layers.Activation('softmax')(attention)
    # attention = layers.RepeatVector(16)(attention)
    # attention = layers.Permute([2, 1], name='attention_vec')(attention)
    # attention_mul = add([layer, attention])
    # out_attention_mul = Flatten()(attention_mul)
    # output = Dense(2, activation='sigmoid')(out_attention_mul)
    # model = Model(inputs=inputs, outputs=output)

    attention_mul = attention_3d_block(layr[len(layr)-1])
    # attention_mul = attention_3d_block(layer6)
    attention_flatten = Flatten()(attention_mul)
    drop2 = Dropout(0.15)(attention_flatten)
    output = Dense(1, activation='sigmoid')(drop2)
    model = Model(inputs=inputs, outputs=output)
    # model = Model(inputs=inputs, outputs=layer8)
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    if tf.test.is_gpu_available():
        print("GPUs available: ")
        print(tf.config.experimental.list_physical_devices('XLA_GPU'))
        print(tf.config.experimental.list_physical_devices('GPU'))
    print("Devices List: ")
    print(device_lib.list_local_devices())
    if device_type=="GPU":
        device_t="/device:GPU:0"
    else:
        device_t="/device:CPU:0"
        
    if classw==True or classwcomp==True:
        class_weight=class_weight
    else: 
        class_weight=None
    with tf.device(device_t):
        history = model.fit(
            x=X_train, y=y_train, 
            epochs=epochs, 
            validation_data=(X_valid,y_valid),
            callbacks=callbacks,
            batch_size=batch_size,
            class_weight=class_weight,
            sample_weight=sample_weight,
            shuffle=shuffle
        )
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    
    print("process time: ", ptime, "real time: ", rtime)
    
    # timestr = time.strftime("%Y%m%d_%H%M%S")
    # print("timestr: ", timestr)
    # results_dir="./results"
    # if not os.path.exists(results_dir): os.mkdir(results_dir)
    # timestr='20210508_150602'
    # model = keras.models.load_model(results_dir+'/'+'deeplearning3tW_d1_model_20210508_150602')
    # history_df=pd.read_csv(model_dir+'/'+'deeplearning3tW_d1_model_20210508_150602_history.csv', index_col=0, header=0)
    model_dir=results_dir+'/model_attgru'
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model.save(model_dir+"/model.h5")
    model.save_weights(model_dir+"/weights.h5")
    orig_stdout = sys.stdout
    f = open(model_dir+"/devices.txt", 'w+')
    sys.stdout = f
    for device in device_lib.list_local_devices():
        print(device)
    sys.stdout = orig_stdout
    f.close()
    mparams_file=model_dir+"/modelparams.txt"
    with open(mparams_file, 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.write("program name: "+sys.argv[0]+ '\n')
        f.write("device used: "+device_t+ '\n')
        f.write("optimizer: "+optimizer_name+ '\n')
        f.write("loss: "+loss+ '\n')
        f.write("metrics: "+str(metrics)+ '\n')
        f.write("patience: "+ str(patience)+ '\n')
        f.write("min_delta: "+str(min_delta)+ '\n')
        f.write("restore_best_weights: "+str(restore_best_weights)+ '\n')
        f.write("gru_unit: "+str(gru_unit)+ '\n')
        # f.write("validation_split: "+str(validation_split)+ '\n')
        # f.write("batch_size: "+str(batch_size)+ '\n')
        # f.write("steps_per_epoch: "+str(steps_per_epoch)+ '\n')
        # f.write("epochs: "+str(epochs)+ '\n')
        # f.write("padding: "+padding+ '\n')
        # f.write("strides: "+str(strides)+ '\n')
        # f.write("kernel_size: "+str(kernel_size)+ '\n')
        # f.write("n_filter: "+str(n_filter)+ '\n')
        # f.write("max_pool: "+str(max_pool)+ '\n')
        f.write("callbacks: "+"[early_stopping]"+ '\n')
        #f.write("class_weight: "+str(class_weight)+ '\n')
        f.write("shuffle: "+str(shuffle)+ '\n')
        f.write("Arguments: "+str(vars(args))+ '\n')
        f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ '\n')
        f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ '\n')
        f.write("Test dataset 355 undersampled (1:1): "+str(under_1_1_test_355)+ '\n')
        f.write("Test dataset 448 undersampled (1:1): "+str(under_1_1_test_448)+ '\n')
        f.write("Training dataset: "+training_dataset+ '\n')
        # f.write("The length of training dataset: "+str(len(df1))+ '\n')
        # f.write("The length of case study(cs) dataset: "+str(len(df_cs))+ '\n')
        # f.write("The length of training dataset w/o cs: "+str(len(df))+ '\n')
        f.write("The number of columns of training dataset w/o label: "+str(len(X_train[0][0]))+ '\n')
        f.write("The size of sliding windows : "+str(len(X_train[0]))+ '\n')
        f.write("The length of training dataset after splitting: "+str(len(X_train))+ '\n')
        f.write("The length of validation dataset: "+str(len(X_valid))+ '\n')
        f.close()
        
    plot_model_history(history,model_dir)
    
    metrics_results=[]
    
    print("Dtestset 355 \n")
    test_dataset_355="df_alldataW_Dset_355.csv"
    for i in [0.536,0.591031]:#np.arange(0.10,0.20,0.01):
        print(i)
        testset_prediction_eval2(model,X_355,y_355,[],model_dir,metrics_results,"X_355",i)
    
    print("Dtestset 448 \n")
    test_dataset_448="df_alldataW_Dset_448.csv"
    testset_prediction_eval(model,X_448,y_448,[],model_dir,metrics_results,"X_448")
    
    with open(mparams_file, 'a') as f:
        f.write("Test dataset: "+test_dataset_355+ '\n')
        f.write("The length of test dataset 355: "+str(len(X_355))+ '\n')
        f.write("Test dataset: "+test_dataset_448+ '\n')
        f.write("The length of test dataset 448: "+str(len(X_448))+ '\n')
        f.close()
    df_metrics_all=pd.concat(metrics_results)
    print(df_metrics_all)
    
    df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
    df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")

def attention_3d_block(inputs):
    input_dim = int(inputs.shape[1])
    a = layers.Permute((2, 1))(inputs)
    a = layers.Dense(input_dim, activation='softmax')(a)
    a_probs = layers.Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = layers.multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

def tuningRNN(results_dir):
    if undersampling==True:
        under_1_1_training=True
        under_1_1_valid=True
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl="_under"
    elif all_data==True:
        under_1_1_training=False
        under_1_1_valid=False
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl="_all"
    else:
        under_1_1_training=False
        under_1_1_valid=False
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl=""
    training_dataset="df_alldataW_all.csv"
    if all_data==True: 
        X_train, y_train, \
        X_355, y_355, \
        X_448, y_448 = load_data(under_lbl)
    else:
        X_train, y_train, X_valid, y_valid, \
        X_355, y_355, \
        X_448, y_448 = load_data(under_lbl)
    
    n_folds = 5
    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold = []
    auc_per_fold = []
    loss_per_fold = []
    precision_per_fold=[]
    recall_per_fold=[]
    aupr_per_fold=[]
    f1_per_fold=[]
    specivity_per_fold=[]
    mcc_per_fold=[]
    mse_per_fold=[]
    mn_mtr_ls=[]
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        if all_data==True:
            train, validation = X_train[train_index], X_train[validation_index]
            target_train, target_val = y_train[train_index], y_train[validation_index]
        else:
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid
        # sm = SMOTE(random_state=42)
        # X_train_res, y_train_res = sm.fit_resample(train, target_train)
        # print (X_train_res.shape, y_train_res.shape)
        if scaler_type=="Standard":
            scaler = StandardScaler()
        elif scaler_type=="MinMax":
            scaler = MinMaxScaler()
        elif scaler_type=="Normalizer":
            scaler = Normalizer(norm='max')
        if scale_data==True:
            for i in range(len(train)):
                train[i]=scaler.fit_transform(train[i])
            for i in range(len(validation)):
                validation[i]=scaler.transform(validation[i])
        class_num=np.unique(target_train)
        tf.keras.backend.clear_session()
        optimizer_name='adam'
        learning_rate=0.002
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss='binary_crossentropy'
        metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000),
                 tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000),
                 'binary_accuracy','mse','Precision','Recall',
                 'TrueNegatives','FalsePositives',
                 'FalseNegatives','TruePositives']
        patience=4
        min_delta=0.001
        restore_best_weights=True
        validation_split=0.2
        batch_size=512
        steps_per_epoch=500
        epochs=20
        epochs_tuner=5
        padding='same'
        strides=(1, 1) 
        kernel_size=5
        n_filter=48
        max_pool=(2,2)
        gru_unit=64
        clsf_model="RNN"
        input_shape=train.shape[1:]
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=restore_best_weights,
            mode='min'
        )
        callbacks=[early_stopping]
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        shuffle=True
        cl_w=class_weight.compute_class_weight(class_weight='balanced',classes=class_num,y=target_train)
        w0=cl_w[0]
        w1=cl_w[1]
        # w0=train.shape[0]/(2*(train.shape[0]-np.sum(target_train,dtype=int)))
        # print(train.shape[0],np.sum(target_train,dtype=int))
        # w1=train.shape[0]/(2*np.sum(target_train,dtype=int))
        classWeight={0: w0, 1: w1}
        #classWeight={0: 0.55, 1: 4.97}
        if sqrtw==True:
        	w0=math.sqrt(w0)
        	w1=math.sqrt(w1)
        if classwcomp==True: 
            classWeight={0: w0, 1: w1}
        else:
            classWeight={0: 0.55, 1: 4.97}
        print(classWeight)
        if sampw==True:
            sample_weight=np.array([w0 if i==0 else w1 for i in target_train.tolist()])
        else:
            sample_weight=None
        
        def build_model(hp):
          model = models.Sequential()
          model.add(layers.Bidirectional(layers.GRU(16,return_sequences=True), 
                                         input_shape=train.shape[1:]))
          model.add(layers.Bidirectional(layers.GRU(16,return_sequences=True), 
                                         input_shape=train.shape[1:]))
          model.add(layers.Bidirectional(layers.GRU(16,return_sequences=True), 
                                         input_shape=train.shape[1:]))
          model.add(layers.Bidirectional(layers.GRU(16,return_sequences=False), 
                                         input_shape=train.shape[1:]))
          # model.add(layers.Flatten())
          model.add(layers.Dense(hp.Choice('units_2', [32, 64]), activation='relu'))
          model.add(layers.Dropout(0.3))
          model.add(layers.Dense(1,activation='sigmoid'))
          model.summary()
          model.compile(
              optimizer=optimizer,
              loss=loss,
              metrics=metrics
          )
          return model
        print('1111111111111111111111111')
        tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=3)
        print('2222222222222222222222222')
        # Define callbacks
        model_dir=results_dir+'/model_rnn_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
        keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
        ]
        if tf.test.is_gpu_available():
            print("GPUs available: ")
            print(tf.config.experimental.list_physical_devices('XLA_GPU'))
            print(tf.config.experimental.list_physical_devices('GPU'))
        print("Devices List: ")
        print(device_lib.list_local_devices())
        if device_type=="GPU":
            device_t="/device:GPU:0"
        else:
            device_t="/device:CPU:0"
            
        if classw==True or classwcomp==True:
            classWeight=classWeight
        else: 
            classWeight=None
        if all_data==True:
            if novald==False:
                validation_data=(validation,target_val)
                callbacks_tuner=callbacks
                callbacks=callbacks_check
                
            else:
                validation_data=None
                callbacks_tuner=None
                callbacks=None
                
        else:
            validation_data=(validation,target_val)
            callbacks_tuner=callbacks
            callbacks=callbacks_check
        print('33333333333333333333333333')
        with tf.device(device_t):
            tuner.search(
                train, target_train, 
                epochs=epochs_tuner, 
                validation_data=validation_data,
                callbacks=callbacks_tuner,
                batch_size=batch_size,
                class_weight=classWeight,
                sample_weight=sample_weight,
                shuffle=shuffle
            )
        print('44444444444444444444444444444')
        print(tuner.get_best_hyperparameters(num_trials=1))
        print(tuner.get_best_hyperparameters(num_trials=2))
        best_hp = tuner.get_best_hyperparameters()[0]
        model = tuner.hypermodel.build(best_hp)
        with tf.device(device_t):
            history = model.fit(
                train, target_train, 
                epochs=epochs, 
                validation_data=validation_data,
                callbacks=callbacks,
                batch_size=batch_size,
                class_weight=classWeight,
                sample_weight=sample_weight,
                shuffle=shuffle
            )
        model.summary()
        scores = model.evaluate(validation, target_val, verbose=0)
        print(f'Validation Scores for fold {fold_no}: {model.metrics_names[0]} : {scores[0]}; {model.metrics_names[1]} : {scores[1]}')
        print(f'                          : {model.metrics_names[2]} : {scores[2]}; {model.metrics_names[3]} : {scores[3]}')
        print(f'                          : {model.metrics_names[4]} : {scores[4]}; {model.metrics_names[5]} : {scores[5]}')
        print(f'                          : {model.metrics_names[6]} : {scores[6]}')
        print(f'                          : {model.metrics_names[7]} : {int(scores[7])}; {model.metrics_names[8]} : {int(scores[8])}')
        print(f'                          : {model.metrics_names[9]} : {int(scores[9])}; {model.metrics_names[10]} : {int(scores[10])}')
        print(model.metrics_names)
        TP=scores[model.metrics_names.index('true_positives')]
        TN=scores[model.metrics_names.index('true_negatives')]
        FP=scores[model.metrics_names.index('false_positives')]
        FN=scores[model.metrics_names.index('false_negatives')]
        try:
            specivity_sc=TN/(TN+FP)
        except:
            specivity_sc=0
        try:
            recall_sc=TP/(TP+FN)
        except:
            recall_sc=0
        try:
            precision_sc=TP/(TP+FP)
        except:
            precision_sc=0
        try:
            f1_sc=2*precision_sc*recall_sc*(precision_sc+recall_sc)
        except:
            f1_sc=0
        try:
            mcc_sc=((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        except:
            mcc_sc=0
        precision_per_fold.append(precision_sc)
        recall_per_fold.append(recall_sc)
        acc_per_fold.append(scores[model.metrics_names.index('binary_accuracy')])
        loss_per_fold.append(scores[model.metrics_names.index('loss')])
        try:
            mse_per_fold.append(scores[model.metrics_names.index('mean_squared_error')])
        except:
            mse_per_fold.append(scores[model.metrics_names.index('mse')])
        auc_per_fold.append(scores[model.metrics_names.index('auc_1')])
        aupr_per_fold.append(scores[model.metrics_names.index('auc')])
        f1_per_fold.append(f1_sc)
        specivity_per_fold.append(specivity_sc)
        mcc_per_fold.append(mcc_sc)

        fold_no = fold_no + 1
        
        # == Average scores ==
        print('------------------------------------------------------------------------')
        print('Scores per fold')
        for i in range(0, len(acc_per_fold)):
          print('------------------------------------------------------------------------')
          print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}')
          print(f'            - Precision: {precision_per_fold[i]} - Recall: {recall_per_fold[i]}')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        
        print(f'>> Loss: {np.mean(loss_per_fold)} (+- {np.std(loss_per_fold)})')
        print(f'>> MSE: {np.mean(mse_per_fold)} (+- {np.std(mse_per_fold)})')
        print(f'>> Sensitivity: {np.mean(recall_per_fold)} (+- {np.std(recall_per_fold)})')
        print(f'>> Specivity: {np.mean(specivity_per_fold)}.{np.mean(specivity_per_fold):.3f} (+- {np.std(specivity_per_fold)})')
        print(f'>> Precision: {np.mean(precision_per_fold)} (+- {np.std(precision_per_fold)})')
        print(f'>> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'>> F1-Score: {np.mean(f1_per_fold)} (+- {np.std(f1_per_fold)})')
        print(f'>> MCC: {np.mean(mcc_per_fold)} (+- {np.std(mcc_per_fold)})')
        print(f'>> AUROC: {np.mean(auc_per_fold)} (+- {np.std(auc_per_fold)})')
        print(f'>> AUPRC: {np.mean(aupr_per_fold)} (+- {np.std(aupr_per_fold)})')
        print('------------------------------------------------------------------------')
        
        model.save(model_dir+'/model.tf')
        model.save_weights(model_dir+"/weights.tf")
        orig_stdout = sys.stdout
        f = open(model_dir+"/devices.txt", 'w+')
        sys.stdout = f
        for device in device_lib.list_local_devices():
            print(device)
        sys.stdout = orig_stdout
        f.close()
        mparams_file=model_dir+"/modelparams.txt"
        with open(mparams_file, 'w+') as f:
            with redirect_stdout(f):
                model.summary()
            f.write("program name: "+sys.argv[0]+ '\n')
            f.write("device used: "+device_t+ '\n')
            f.write("optimizer: "+optimizer_name+ '\n')
            f.write("loss: "+loss+ '\n')
            f.write("metrics: "+ str(metrics)+ '\n')
            f.write("patience: "+ str(patience)+ '\n')
            f.write("min_delta: "+str(min_delta)+ '\n')
            f.write("restore_best_weights: "+str(restore_best_weights)+ '\n')
            f.write("gru_unit: "+str(gru_unit)+ '\n')
            # f.write("validation_split: "+str(validation_split)+ '\n')
            f.write("batch_size: "+str(batch_size)+ '\n')
            # f.write("steps_per_epoch: "+str(steps_per_epoch)+ '\n')
            f.write("epochs: "+str(epochs)+ '\n')
            f.write("padding: "+padding+ '\n')
            f.write("strides: "+str(strides)+ '\n')
            f.write("n_filter: "+str(n_filter)+ '\n')
            f.write("callbacks: "+"[early_stopping]"+ '\n')
            f.write("classWeight: "+str(classWeight)+ '\n')
            f.write("shuffle: "+str(shuffle)+ '\n')
            f.write("Arguments: "+str(vars(args))+ '\n')
            f.write("Scaling: "+str(scale_data)+ '\n')
            f.write("Scaler: "+str(scaler)+ '\n')
            f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ '\n')
            f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ '\n')
            f.write("Test dataset 355 undersampled (1:1): "+str(under_1_1_test_355)+ '\n')
            f.write("Test dataset 448 undersampled (1:1): "+str(under_1_1_test_448)+ '\n')
            f.write("Training dataset: "+training_dataset+ '\n')
            # f.write("The length of training dataset: "+str(len(df1))+ '\n')
            # f.write("The length of case study(cs) dataset: "+str(len(df_cs))+ '\n')
            # f.write("The length of training dataset w/o cs: "+str(len(df))+ '\n')
            f.write("The number of columns of training dataset w/o label: "+str(train.shape[1])+ '\n')
            f.write("The size of sliding windows : "+str(train.shape[2])+ '\n')
            f.write("The length of training dataset after splitting: "+str(train.shape[0])+ '\n')
            f.write("The length of validation dataset: "+str(len(validation))+ '\n')
            f.close()
            
        plot_model_history(history,model_dir)
        
        metrics_results=[]
                
        print("Dtestset 355 \n")
        test_dataset_355="df_alldataW_Dset_355.csv"
        testset_prediction_eval(model,X_355,y_355,[],model_dir,metrics_results,"X_355",scale_data,scaler,clsf_model)
        
        print("Dtestset 448 \n")
        test_dataset_448="df_alldataW_Dset_448.csv"
        testset_prediction_eval(model,X_448,y_448,[],model_dir,metrics_results,"X_448",scale_data,scaler,clsf_model)
        
        with open(mparams_file, 'a') as f:
            f.write("Test dataset: "+test_dataset_355+ '\n')
            f.write("The length of test dataset 355: "+str(len(X_355))+ '\n')
            f.write("Test dataset: "+test_dataset_448+ '\n')
            f.write("The length of test dataset 448: "+str(len(X_448))+ '\n')
            f.close()
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        test_mtr=["X_355","X_448"]
        test_mtr=[i+"_1" for i in test_mtr]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')["AUROC"].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        if all_data==False or novald==False: break
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print(mn_mtr_ls)
    print(mn_mtr_ls.index(max(mn_mtr_ls)))
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/rnn.dir', 'w') as f:
        f.write(results_dir+f'/model_rnn_fold/fold_{fd_slct}/fold_{fd_slct}.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)

def LSTMs2s(results_dir):
    if undersampling==True:
        under_1_1_training=True
        under_1_1_valid=True
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl="_down"
    else:
        under_1_1_training=False
        under_1_1_valid=False
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl=""
    training_dataset="df_alldataW_all.csv"
    X_train, y_train, X_valid, y_valid, \
    X_355, y_355, \
    X_448, y_448 = load_data(under_lbl)
    
    tf.keras.backend.clear_session()
    optimizer_name='adam'
    learning_rate=0.002
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss='binary_crossentropy'
    metrics=[tf.keras.metrics.AUC(curve='PR')]
    patience=4
    min_delta=0.001
    restore_best_weights=True
    validation_split=0.2
    batch_size=512
    steps_per_epoch=500
    epochs=10
    padding='same'
    strides=(1, 1) 
    kernel_size=5
    n_filter=48
    max_pool=(2,2)
    gru_unit=16
    TIME_STEPS = 2
    INPUT_DIM = 18
    lstm_units = 16
    nb_classes = 1
    num_layr=60
    input_shape=X_train.shape[1:]
    early_stopping = keras.callbacks.EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights
    )
    callbacks=[early_stopping]
    w0=X_train.shape[0]/(2*(X_train.shape[0]-np.sum(y_train,dtype=int)))
    w1=X_train.shape[0]/(2*np.sum(y_train,dtype=int))
    class_weight={0: w0, 1: w1}
    #class_weight={0: 0.55, 1: 4.97}
    if sqrtw==True:
    	w0=math.sqrt(w0)
    	w1=math.sqrt(w1)
    if classwcomp==True: 
        class_weight={0: w0, 1: w1}
    else:
        class_weight={0: 0.55, 1: 4.97}
    print(class_weight)
    if sampw==True:
        sample_weight=np.array([w0 if i==0 else w1 for i in y_train.tolist()])
    else:
        sample_weight=None
    shuffle=True
    print(X_train.shape[0],X_train.shape[1],X_train.shape[2])
    
    start_time = time.time()
    start = time.process_time()
    
    """
    ## Build the model
    """

    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=X_train.shape[1:])
    encoder = layers.LSTM(lstm_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    
    # Set up the decoder, using `encoder_states` as initial state.

    decoder_inputs = keras.Input(shape=(None,1))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(1, activation="sigmoid")
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    """
    ## Train the model
    """
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    tf.keras.utils.plot_model_history(model, to_file=results_dir+"/model.png", show_shapes=True)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    if tf.test.is_gpu_available():
        print("GPUs available: ")
        print(tf.config.experimental.list_physical_devices('XLA_GPU'))
        print(tf.config.experimental.list_physical_devices('GPU'))
    print("Devices List: ")
    print(device_lib.list_local_devices())
    if device_type=="GPU":
        device_t="/device:GPU:0"
    else:
        device_t="/device:CPU:0"
        
    if classw==True or classwcomp==True:
        class_weight=class_weight
    else: 
        class_weight=None
    with tf.device(device_t):
        history = model.fit(
            x=[X_train, y_train], y=y_train, 
            epochs=epochs, 
            validation_data=([X_valid, y_valid],y_valid),
            callbacks=callbacks,
            batch_size=batch_size,
            class_weight=class_weight,
            sample_weight=sample_weight,
            shuffle=shuffle
        )
        
    # Save model
    model.save("s2s")
    model = keras.models.load_model("s2s")
    
    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = keras.Input(shape=(lstm_units,))
    decoder_state_input_c = keras.Input(shape=(lstm_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )
    
    X_valid=encoder_model.predict(X_valid)
    output_tokens, h, c = decoder_model.predict([y_valid] + X_valid)
    # print(output_tokens)
    labels = (output_tokens >= 0.5).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    print(classification_report(y_valid, y_pred))
    print(confusion_matrix(y_valid,y_pred))
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    
    print("process time: ", ptime, "real time: ", rtime)
    
    # timestr = time.strftime("%Y%m%d_%H%M%S")
    # print("timestr: ", timestr)
    # results_dir="./results"
    # if not os.path.exists(results_dir): os.mkdir(results_dir)
    # timestr='20210508_150602'
    # model = keras.models.load_model(results_dir+'/'+'deeplearning3tW_d1_model_20210508_150602')
    # history_df=pd.read_csv(model_dir+'/'+'deeplearning3tW_d1_model_20210508_150602_history.csv', index_col=0, header=0)
    model_dir=results_dir+'/model_ensb'
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model.save(model_dir+"/model.h5")
    model.save_weights(model_dir+"/weights.h5")
    orig_stdout = sys.stdout
    f = open(model_dir+"/devices.txt", 'w+')
    sys.stdout = f
    for device in device_lib.list_local_devices():
        print(device)
    sys.stdout = orig_stdout
    f.close()
    mparams_file=model_dir+"/modelparams.txt"
    with open(mparams_file, 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.write("program name: "+sys.argv[0]+ '\n')
        f.write("device used: "+device_t+ '\n')
        f.write("optimizer: "+optimizer_name+ '\n')
        f.write("loss: "+loss+ '\n')
        f.write("metrics: "+str(metrics)+ '\n')
        f.write("patience: "+ str(patience)+ '\n')
        f.write("min_delta: "+str(min_delta)+ '\n')
        f.write("restore_best_weights: "+str(restore_best_weights)+ '\n')
        f.write("gru_unit: "+str(gru_unit)+ '\n')
        # f.write("validation_split: "+str(validation_split)+ '\n')
        f.write("batch_size: "+str(batch_size)+ '\n')
        # f.write("steps_per_epoch: "+str(steps_per_epoch)+ '\n')
        f.write("epochs: "+str(epochs)+ '\n')
        # f.write("padding: "+padding+ '\n')
        # f.write("strides: "+str(strides)+ '\n')
        # f.write("kernel_size: "+str(kernel_size)+ '\n')
        # f.write("n_filter: "+str(n_filter)+ '\n')
        # f.write("max_pool: "+str(max_pool)+ '\n')
        f.write("callbacks: "+"[early_stopping]"+ '\n')
        f.write("class_weight: "+str(class_weight)+ '\n')
        f.write("shuffle: "+str(shuffle)+ '\n')
        f.write("Arguments: "+str(vars(args))+ '\n')
        f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ '\n')
        f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ '\n')
        f.write("Test dataset 355 undersampled (1:1): "+str(under_1_1_test_355)+ '\n')
        f.write("Test dataset 448 undersampled (1:1): "+str(under_1_1_test_448)+ '\n')
        f.write("Training dataset: "+training_dataset+ '\n')
        # f.write("The length of training dataset: "+str(len(df1))+ '\n')
        # f.write("The length of case study(cs) dataset: "+str(len(df_cs))+ '\n')
        # f.write("The length of training dataset w/o cs: "+str(len(df))+ '\n')
        f.write("The number of columns of training dataset w/o label: "+str(len(X_train[0][0]))+ '\n')
        f.write("The size of sliding windows : "+str(len(X_train[0]))+ '\n')
        f.write("The length of training dataset after splitting: "+str(len(X_train))+ '\n')
        f.write("The length of validation dataset: "+str(len(X_valid))+ '\n')
        f.close()
        
    plot_model_history(history,model_dir)
    
    metrics_results=[]
    clsf_model='LSTM'
    scaler="Standart"
    scale_data=False
    print("Dtestset 355 \n")
    X_355=encoder_model.predict(X_355)
    output_tokens, h, c = decoder_model.predict([y_355] + X_355)
    test_dataset_355="df_alldataW_Dset_355.csv"
    labels = (output_tokens >= 0.5).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    print(classification_report(y_355, y_pred))
    print(confusion_matrix(y_355,y_pred))
    #testset_prediction_eval(model,X_355,output_tokens,[],model_dir,metrics_results,"X_355",scale_data,scaler,clsf_model)
    
    print("Dtestset 448 \n")
    X_448=encoder_model.predict(X_448)
    output_tokens, h, c = decoder_model.predict([y_448] + X_448)
    print(output_tokens.shape)
    test_dataset_448="df_alldataW_Dset_448.csv"
    labels = (output_tokens >= 0.5).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    print(classification_report(y_448, y_pred))
    print(confusion_matrix(y_448,y_pred))
    #testset_prediction_eval(model,X_448,y_448,[],model_dir,metrics_results,"X_448")
    # for i in output_tokens:
    #     if i>0.5: print(i,end=' ')
    with open(mparams_file, 'a') as f:
        f.write("Test dataset: "+test_dataset_355+ '\n')
        f.write("The length of test dataset 355: "+str(len(X_355))+ '\n')
        f.write("Test dataset: "+test_dataset_448+ '\n')
        f.write("The length of test dataset 448: "+str(len(X_448))+ '\n')
        f.close()
    # df_metrics_all=pd.concat(metrics_results)
    # print(df_metrics_all)
    
    # df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
    # df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")

def LSTMs2s2(results_dir):
    if undersampling==True:
        under_1_1_training=True
        under_1_1_valid=True
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl="_down"
    else:
        under_1_1_training=False
        under_1_1_valid=False
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl=""
    training_dataset="df_alldataW_all.csv"
    X_train, y_train, X_valid, y_valid, \
    X_355, y_355, \
    X_448, y_448 = load_data(under_lbl)
    
    tf.keras.backend.clear_session()
    optimizer_name='adam'
    learning_rate=0.002
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss='binary_crossentropy'
    metrics=[tf.keras.metrics.AUC(curve='PR')]
    patience=4
    min_delta=0.001
    restore_best_weights=True
    validation_split=0.2
    batch_size=512
    steps_per_epoch=500
    epochs=15
    padding='same'
    strides=(1, 1) 
    kernel_size=5
    n_filter=48
    max_pool=(2,2)
    gru_unit=16
    TIME_STEPS = 2
    INPUT_DIM = 18
    lstm_units = 16
    nb_classes = 1
    num_layr=60
    start_input=-10
    input_shape=X_train.shape[1:]
    early_stopping = keras.callbacks.EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights
    )
    callbacks=[early_stopping]
    w0=X_train.shape[0]/(2*(X_train.shape[0]-np.sum(y_train,dtype=int)))
    w1=X_train.shape[0]/(2*np.sum(y_train,dtype=int))
    class_weight={0: w0, 1: w1}
    #class_weight={0: 0.55, 1: 4.97}
    if sqrtw==True:
    	w0=math.sqrt(w0)
    	w1=math.sqrt(w1)
    if classwcomp==True: 
        class_weight={0: w0, 1: w1}
    else:
        class_weight={0: 0.55, 1: 4.97}
    print(class_weight)
    if sampw==True:
        sample_weight=np.array([w0 if i==0 else w1 for i in y_train.tolist()])
    else:
        sample_weight=None
    shuffle=True
    print(X_train.shape[0],X_train.shape[1],X_train.shape[2])
    
    start_time = time.time()
    start = time.process_time()
    
    """
    ## Build the model
    """

    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=X_train.shape[1:])
    encoder = layers.LSTM(lstm_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    
    # Set up the decoder, using `encoder_states` as initial state.

    decoder_inputs = keras.Input(shape=(None,1))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(1, activation="sigmoid")
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    print(model.layers)
    """
    ## Train the model
    """
    train_decoder_input=np.full((y_train.shape[0],), start_input)
    valid_decoder_input=np.full((y_valid.shape[0],), start_input)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    tf.keras.utils.plot_model_history(model, to_file=results_dir+"/model.png", show_shapes=True)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    if tf.test.is_gpu_available():
        print("GPUs available: ")
        print(tf.config.experimental.list_physical_devices('XLA_GPU'))
        print(tf.config.experimental.list_physical_devices('GPU'))
    print("Devices List: ")
    print(device_lib.list_local_devices())
    if device_type=="GPU":
        device_t="/device:GPU:0"
    else:
        device_t="/device:CPU:0"
        
    if classw==True or classwcomp==True:
        class_weight=class_weight
    else: 
        class_weight=None
    with tf.device(device_t):
        history = model.fit(
            x=[X_train, train_decoder_input], y=y_train, 
            epochs=epochs, 
            validation_data=([X_valid, valid_decoder_input],y_valid),
            callbacks=callbacks,
            batch_size=batch_size,
            class_weight=class_weight,
            sample_weight=sample_weight,
            shuffle=shuffle
        )
        
    # Save model
    model.save("s2s")
    model = keras.models.load_model("s2s")
    
    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = keras.Input(shape=(lstm_units,))
    decoder_state_input_c = keras.Input(shape=(lstm_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )
    
    X_valid=encoder_model.predict(X_valid)
    output_tokens, h, c = decoder_model.predict([valid_decoder_input] + X_valid)
    # print(output_tokens)
    labels = (output_tokens >= 0.5).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    print(classification_report(y_valid, y_pred))
    print(confusion_matrix(y_valid,y_pred))
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    
    print("process time: ", ptime, "real time: ", rtime)
    
    # timestr = time.strftime("%Y%m%d_%H%M%S")
    # print("timestr: ", timestr)
    # results_dir="./results"
    # if not os.path.exists(results_dir): os.mkdir(results_dir)
    # timestr='20210508_150602'
    # model = keras.models.load_model(results_dir+'/'+'deeplearning3tW_d1_model_20210508_150602')
    # history_df=pd.read_csv(model_dir+'/'+'deeplearning3tW_d1_model_20210508_150602_history.csv', index_col=0, header=0)
    model_dir=results_dir+'/model_ensb'
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model.save(model_dir+"/model.h5")
    model.save_weights(model_dir+"/weights.h5")
    orig_stdout = sys.stdout
    f = open(model_dir+"/devices.txt", 'w+')
    sys.stdout = f
    for device in device_lib.list_local_devices():
        print(device)
    sys.stdout = orig_stdout
    f.close()
    mparams_file=model_dir+"/modelparams.txt"
    with open(mparams_file, 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.write("program name: "+sys.argv[0]+ '\n')
        f.write("device used: "+device_t+ '\n')
        f.write("optimizer: "+optimizer_name+ '\n')
        f.write("loss: "+loss+ '\n')
        f.write("metrics: "+str(metrics)+ '\n')
        f.write("patience: "+ str(patience)+ '\n')
        f.write("min_delta: "+str(min_delta)+ '\n')
        f.write("restore_best_weights: "+str(restore_best_weights)+ '\n')
        f.write("gru_unit: "+str(gru_unit)+ '\n')
        # f.write("validation_split: "+str(validation_split)+ '\n')
        f.write("batch_size: "+str(batch_size)+ '\n')
        # f.write("steps_per_epoch: "+str(steps_per_epoch)+ '\n')
        f.write("epochs: "+str(epochs)+ '\n')
        # f.write("padding: "+padding+ '\n')
        # f.write("strides: "+str(strides)+ '\n')
        # f.write("kernel_size: "+str(kernel_size)+ '\n')
        # f.write("n_filter: "+str(n_filter)+ '\n')
        # f.write("max_pool: "+str(max_pool)+ '\n')
        f.write("callbacks: "+"[early_stopping]"+ '\n')
        f.write("class_weight: "+str(class_weight)+ '\n')
        f.write("shuffle: "+str(shuffle)+ '\n')
        f.write("Arguments: "+str(vars(args))+ '\n')
        f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ '\n')
        f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ '\n')
        f.write("Test dataset 355 undersampled (1:1): "+str(under_1_1_test_355)+ '\n')
        f.write("Test dataset 448 undersampled (1:1): "+str(under_1_1_test_448)+ '\n')
        f.write("Training dataset: "+training_dataset+ '\n')
        # f.write("The length of training dataset: "+str(len(df1))+ '\n')
        # f.write("The length of case study(cs) dataset: "+str(len(df_cs))+ '\n')
        # f.write("The length of training dataset w/o cs: "+str(len(df))+ '\n')
        f.write("The number of columns of training dataset w/o label: "+str(len(X_train[0][0]))+ '\n')
        f.write("The size of sliding windows : "+str(len(X_train[0]))+ '\n')
        f.write("The length of training dataset after splitting: "+str(len(X_train))+ '\n')
        f.write("The length of validation dataset: "+str(len(X_valid))+ '\n')
        f.close()
        
    plot_model_history(history,model_dir)
    
    metrics_results=[]
    clsf_model='LSTM'
    scaler="Standart"
    scale_data=False
    print("Dtestset 355 \n")
    t_355_decoder_input=np.full((y_355.shape[0],), start_input)
    X_355=encoder_model.predict(X_355)
    output_tokens, h, c = decoder_model.predict([t_355_decoder_input] + X_355)
    test_dataset_355="df_alldataW_Dset_355.csv"
    labels = (output_tokens >= 0.5).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    print(classification_report(y_355, y_pred))
    print(confusion_matrix(y_355,y_pred))
    testset_prediction_eval3(decoder_model,[t_355_decoder_input] + X_355,y_355,[],model_dir,metrics_results,"X_355",scale_data,scaler,clsf_model)
    
    print("Dtestset 448 \n")
    t_448_decoder_input=np.full((y_448.shape[0],), start_input)
    X_448=encoder_model.predict(X_448)
    output_tokens, h, c = decoder_model.predict([t_448_decoder_input] + X_448)
    print(output_tokens.shape)
    test_dataset_448="df_alldataW_Dset_448.csv"
    labels = (output_tokens >= 0.5).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    print(classification_report(y_448, y_pred))
    print(confusion_matrix(y_448,y_pred))
    testset_prediction_eval3(decoder_model,[t_448_decoder_input] + X_448,y_448,[],model_dir,metrics_results,"X_448",scale_data,scaler,clsf_model)
    # for i in output_tokens:
    #     if i>0.5: print(i,end=' ')
    with open(mparams_file, 'a') as f:
        f.write("Test dataset: "+test_dataset_355+ '\n')
        f.write("The length of test dataset 355: "+str(len(X_355))+ '\n')
        f.write("Test dataset: "+test_dataset_448+ '\n')
        f.write("The length of test dataset 448: "+str(len(X_448))+ '\n')
        f.close()
    df_metrics_all=pd.concat(metrics_results)
    print(df_metrics_all)
    
    df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
    df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
    
def GRUs2s(results_dir):
    if undersampling==True:
        under_1_1_training=True
        under_1_1_valid=True
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl="_down"
    else:
        under_1_1_training=False
        under_1_1_valid=False
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl=""
    training_dataset="df_alldataW_all.csv"
    X_train, y_train, X_valid, y_valid, \
    X_355, y_355, \
    X_448, y_448 = load_data(under_lbl)
    
    tf.keras.backend.clear_session()
    optimizer_name='adam'
    learning_rate=0.002
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss='binary_crossentropy'
    metrics=[tf.keras.metrics.AUC(curve='PR')]
    patience=4
    min_delta=0.001
    restore_best_weights=True
    validation_split=0.2
    batch_size=512
    steps_per_epoch=500
    epochs=15
    padding='same'
    strides=(1, 1) 
    kernel_size=5
    n_filter=48
    max_pool=(2,2)
    gru_unit=16
    TIME_STEPS = 2
    INPUT_DIM = 18
    lstm_units = 16
    nb_classes = 1
    num_layr=60
    start_input=-10
    input_shape=X_train.shape[1:]
    early_stopping = keras.callbacks.EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights
    )
    callbacks=[early_stopping]
    w0=X_train.shape[0]/(2*(X_train.shape[0]-np.sum(y_train,dtype=int)))
    w1=X_train.shape[0]/(2*np.sum(y_train,dtype=int))
    class_weight={0: w0, 1: w1}
    #class_weight={0: 0.55, 1: 4.97}
    if sqrtw==True:
    	w0=math.sqrt(w0)
    	w1=math.sqrt(w1)
    if classwcomp==True: 
        class_weight={0: w0, 1: w1}
    else:
        class_weight={0: 0.55, 1: 4.97}
    print(class_weight)
    if sampw==True:
        sample_weight=np.array([w0 if i==0 else w1 for i in y_train.tolist()])
    else:
        sample_weight=None
    shuffle=True
    print(X_train.shape[0],X_train.shape[1],X_train.shape[2])
    
    start_time = time.time()
    start = time.process_time()
    
    """
    ## Build the model
    """

    # Define an input sequence and process it.
    encoder_inputs = tf.keras.Input(shape=X_train.shape[1:])
    encoder = layers.GRU(lstm_units, return_state=True)
    encoder_outputs, state_h = encoder(encoder_inputs)
    
    # We discard `encoder_outputs` and only keep the states.
    # encoder_states = [state_h]
    
    # Set up the decoder, using `encoder_states` as initial state.

    decoder_inputs = tf.keras.Input(shape=(None,1))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = layers.GRU(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _= decoder_lstm(decoder_inputs, initial_state=state_h)
    decoder_dense = layers.Dense(1, activation="sigmoid")
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    """
    ## Train the model
    """
    train_decoder_input=np.full((y_train.shape[0],), start_input)
    valid_decoder_input=np.full((y_valid.shape[0],), start_input)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    tf.keras.utils.plot_model_history(model, to_file=results_dir+"/model.png", show_shapes=True)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    if tf.test.is_gpu_available():
        print("GPUs available: ")
        print(tf.config.experimental.list_physical_devices('XLA_GPU'))
        print(tf.config.experimental.list_physical_devices('GPU'))
    print("Devices List: ")
    print(device_lib.list_local_devices())
    if device_type=="GPU":
        device_t="/device:GPU:0"
    else:
        device_t="/device:CPU:0"
        
    if classw==True or classwcomp==True:
        class_weight=class_weight
    else: 
        class_weight=None
    with tf.device(device_t):
        history = model.fit(
            x=[X_train, train_decoder_input], y=y_train, 
            epochs=epochs, 
            validation_data=([X_valid, valid_decoder_input],y_valid),
            callbacks=callbacks,
            batch_size=batch_size,
            class_weight=class_weight,
            sample_weight=sample_weight,
            shuffle=shuffle
        )
        
    # Save model
    model.save("s2s")
    model = tf.keras.models.load_model("s2s")
    tf.keras.utils.plot_model(model, to_file=results_dir+"/model_1.png", show_shapes=True)
    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc = model.layers[2].output  # lstm_1
    # encoder_states = [state_h_enc]
    encoder_model = tf.keras.Model(encoder_inputs, state_h_enc)
    tf.keras.utils.plot_model(encoder_model, to_file=results_dir+"/model_enc.png", show_shapes=True)
    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = tf.keras.Input(shape=(lstm_units,))
    # decoder_state_input_c = tf.keras.Input(shape=(lstm_units,))
    # decoder_states_inputs = [decoder_state_input_h]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_state_input_h
    )
    # decoder_states = [state_h_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.Model(
       [decoder_inputs] + [decoder_state_input_h] , [decoder_outputs] + [state_h_dec] 
    )
    tf.keras.utils.plot_model(decoder_model, to_file=results_dir+"/model_dec.png", show_shapes=True)
    X_valid=encoder_model.predict(X_valid)    
    output_tokens, h= decoder_model.predict([valid_decoder_input] + [X_valid])
    # print(output_tokens)
    labels = (output_tokens >= 0.5).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    print(classification_report(y_valid, y_pred))
    print(confusion_matrix(y_valid,y_pred))
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    
    print("process time: ", ptime, "real time: ", rtime)
    
    # timestr = time.strftime("%Y%m%d_%H%M%S")
    # print("timestr: ", timestr)
    # results_dir="./results"
    # if not os.path.exists(results_dir): os.mkdir(results_dir)
    # timestr='20210508_150602'
    # model = keras.models.load_model(results_dir+'/'+'deeplearning3tW_d1_model_20210508_150602')
    # history_df=pd.read_csv(model_dir+'/'+'deeplearning3tW_d1_model_20210508_150602_history.csv', index_col=0, header=0)
    model_dir=results_dir+'/model_ensb'
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    decoder_model.save(model_dir+"/model.h5")
    decoder_model.save_weights(model_dir+"/weights.h5")
    orig_stdout = sys.stdout
    f = open(model_dir+"/devices.txt", 'w+')
    sys.stdout = f
    for device in device_lib.list_local_devices():
        print(device)
    sys.stdout = orig_stdout
    f.close()
    mparams_file=model_dir+"/modelparams.txt"
    with open(mparams_file, 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.write("program name: "+sys.argv[0]+ '\n')
        f.write("device used: "+device_t+ '\n')
        f.write("optimizer: "+optimizer_name+ '\n')
        f.write("loss: "+loss+ '\n')
        f.write("metrics: "+str(metrics)+ '\n')
        f.write("patience: "+ str(patience)+ '\n')
        f.write("min_delta: "+str(min_delta)+ '\n')
        f.write("restore_best_weights: "+str(restore_best_weights)+ '\n')
        f.write("gru_unit: "+str(gru_unit)+ '\n')
        # f.write("validation_split: "+str(validation_split)+ '\n')
        f.write("batch_size: "+str(batch_size)+ '\n')
        # f.write("steps_per_epoch: "+str(steps_per_epoch)+ '\n')
        f.write("epochs: "+str(epochs)+ '\n')
        # f.write("padding: "+padding+ '\n')
        # f.write("strides: "+str(strides)+ '\n')
        # f.write("kernel_size: "+str(kernel_size)+ '\n')
        # f.write("n_filter: "+str(n_filter)+ '\n')
        # f.write("max_pool: "+str(max_pool)+ '\n')
        f.write("callbacks: "+"[early_stopping]"+ '\n')
        f.write("class_weight: "+str(class_weight)+ '\n')
        f.write("shuffle: "+str(shuffle)+ '\n')
        f.write("Arguments: "+str(vars(args))+ '\n')
        f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ '\n')
        f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ '\n')
        f.write("Test dataset 355 undersampled (1:1): "+str(under_1_1_test_355)+ '\n')
        f.write("Test dataset 448 undersampled (1:1): "+str(under_1_1_test_448)+ '\n')
        f.write("Training dataset: "+training_dataset+ '\n')
        # f.write("The length of training dataset: "+str(len(df1))+ '\n')
        # f.write("The length of case study(cs) dataset: "+str(len(df_cs))+ '\n')
        # f.write("The length of training dataset w/o cs: "+str(len(df))+ '\n')
        f.write("The number of columns of training dataset w/o label: "+str(X_train.shape[1])+ '\n')
        f.write("The size of sliding windows : "+str(X_train.shape[2])+ '\n')
        f.write("The length of training dataset after splitting: "+str(X_train.shape[0])+ '\n')
        f.write("The length of validation dataset: "+str(len(X_valid))+ '\n')
        f.close()
        
    plot_model_history(history,model_dir)
    
    metrics_results=[]
    clsf_model='LSTM'
    scaler="Standart"
    scale_data=False
    print("Dtestset 355 \n")
    t_355_decoder_input=np.full((y_355.shape[0],), start_input)
    X_355=encoder_model.predict(X_355)
    output_tokens, h= decoder_model.predict([t_355_decoder_input] + [X_355])
    test_dataset_355="df_alldataW_Dset_355.csv"
    labels = (output_tokens >= 0.5).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    print(classification_report(y_355, y_pred))
    print(confusion_matrix(y_355,y_pred))
    testset_prediction_eval4(decoder_model,[t_355_decoder_input] + [X_355],y_355,[],model_dir,metrics_results,"X_355",scale_data,scaler,clsf_model)
    
    print("Dtestset 448 \n")
    t_448_decoder_input=np.full((y_448.shape[0],), start_input)
    X_448=encoder_model.predict(X_448)
    output_tokens, h= decoder_model.predict([t_448_decoder_input] + [X_448])
    test_dataset_448="df_alldataW_Dset_448.csv"
    labels = (output_tokens >= 0.5).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    print(classification_report(y_448, y_pred))
    print(confusion_matrix(y_448,y_pred))
    testset_prediction_eval4(decoder_model,[t_448_decoder_input] + [X_448],y_448,[],model_dir,metrics_results,"X_448",scale_data,scaler,clsf_model)
    # for i in output_tokens:
    #     if i>0.5: print(i,end=' ')
    with open(mparams_file, 'a') as f:
        f.write("Test dataset: "+test_dataset_355+ '\n')
        f.write("The length of test dataset 355: "+str(len(X_355))+ '\n')
        f.write("Test dataset: "+test_dataset_448+ '\n')
        f.write("The length of test dataset 448: "+str(len(X_448))+ '\n')
        f.close()
    df_metrics_all=pd.concat(metrics_results)
    print(df_metrics_all)
    
    df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
    df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")

# Add attention layer to the deep learning network
class attention():
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context
    
class LuongAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(LuongAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    print('\n******* Luong Attention  STARTS******')
    print('query (decoder hidden state): (batch_size, hidden size) ', query.shape)
    print('values (encoder all hidden state): (batch_size, max_len, hidden size) ', values.shape)

    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)
    
    print('query_with_time_axis:(batch_size, 1, hidden size) ', query_with_time_axis.shape)


    values_transposed = tf.transpose(values, perm=[0, 2, 1])
    print('values_transposed:(batch_size, hidden size, max_len) ', values_transposed.shape)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    #BAHDANAU ADDITIVE:
    #score = self.V(tf.nn.tanh(
    #    self.W1(query_with_time_axis) + self.W2(values)))
    
    #LUONGH Dot-product
    score = tf.transpose(tf.matmul(query_with_time_axis, values_transposed) , perm=[0, 2, 1])

    print('score: (batch_size, max_length, 1) ',score.shape)
    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)
    print('attention_weights: (batch_size, max_length, 1) ',attention_weights.shape)
    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    print('context_vector before reduce_sum: (batch_size, max_length, hidden_size) ',context_vector.shape)
    context_vector = tf.reduce_sum(context_vector, axis=1)
    print('context_vector after reduce_sum: (batch_size, hidden_size) ',context_vector.shape)


    print('\n******* Luong Attention ENDS******')
    return context_vector, attention_weights

def LSTMs2sAttention(results_dir):
    if undersampling==True:
        under_1_1_training=True
        under_1_1_valid=True
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl="_down"
    else:
        under_1_1_training=False
        under_1_1_valid=False
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl=""
    training_dataset="df_alldataW_all.csv"
    X_train, y_train, X_valid, y_valid, \
    X_355, y_355, \
    X_448, y_448 = load_data(under_lbl)
    
    tf.keras.backend.clear_session()
    optimizer_name='adam'
    learning_rate=0.0005 #0.002
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss='binary_crossentropy'
    metrics=[tf.keras.metrics.AUC(curve='PR')]
    patience=5
    min_delta=1e-5
    restore_best_weights=True
    validation_split=0.2
    batch_size=512
    steps_per_epoch=500
    epochs=15
    padding='same'
    strides=(1, 1) 
    kernel_size=5
    n_filter=48
    max_pool=(2,2)
    gru_unit=16
    TIME_STEPS = 2
    INPUT_DIM = 18
    lstm_units = 16
    nb_classes = 1
    num_layr=60
    start_input=-2
    input_shape=X_train.shape[1:]
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights,
        mode='min'
    )
    callbacks=[early_stopping]
    w0=X_train.shape[0]/(2*(X_train.shape[0]-np.sum(y_train,dtype=int)))
    w1=X_train.shape[0]/(2*np.sum(y_train,dtype=int))
    class_weight={0: w0, 1: w1}
    #class_weight={0: 0.55, 1: 4.97}
    if sqrtw==True:
    	w0=math.sqrt(w0)
    	w1=math.sqrt(w1)
    if classwcomp==True: 
        class_weight={0: w0, 1: w1}
    else:
        class_weight={0: 0.55, 1: 4.97}
    print(class_weight)
    if sampw==True:
        sample_weight=np.array([w0 if i==0 else w1 for i in y_train.tolist()])
    else:
        sample_weight=None
    shuffle=True
    print(X_train.shape[0],X_train.shape[1],X_train.shape[2])
    
    start_time = time.time()
    start = time.process_time()
    
    """
    ## Build the model
    """

    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=X_train.shape[1:])
    
    # Bidirectional lstm layer
    enc_lstm1 = layers.Bidirectional(layers.LSTM(lstm_units,return_sequences=True,return_state=False))
    encoder_outputs1 = enc_lstm1(encoder_inputs)
    
    enc_lstm2 = layers.Bidirectional(layers.LSTM(lstm_units,return_sequences=True,return_state=False))
    encoder_outputs2 = enc_lstm2(encoder_outputs1)
    
    enc_lstm3 = layers.Bidirectional(layers.LSTM(lstm_units,return_sequences=True,return_state=True))
    encoder_outputs3, forw_state_h, forw_state_c, back_state_h, back_state_c = enc_lstm3(encoder_outputs2)
    
    final_enc_h = layers.Concatenate()([forw_state_h,back_state_h])
    final_enc_c = layers.Concatenate()([forw_state_c,back_state_c])
    
    encoder_states =[final_enc_h, final_enc_c]
    
    # Set up the decoder, using `encoder_states` as initial state.

    decoder_inputs = keras.Input(shape=(None,1))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = layers.LSTM(2*lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    
    #Attention Layer
    # attention_layer = tf.keras.layers.Attention()
    attention_layer = AttentionLayer()
    attention_result,_= attention_layer([encoder_outputs3,decoder_outputs])
    
    # Concat attention output and decoder LSTM output 
    decoder_concat_input = layers.Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_result])
    
    decoder_dense = keras.layers.Dense(1, activation="sigmoid")
    decoder_outputs = decoder_dense(decoder_concat_input)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    """
    ## Train the model
    """
    train_decoder_input=np.full((y_train.shape[0],), start_input)
    valid_decoder_input=np.full((y_valid.shape[0],), start_input)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    tf.keras.utils.plot_model(model, to_file=results_dir+"/model.png", show_shapes=True)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    if tf.test.is_gpu_available():
        print("GPUs available: ")
        print(tf.config.experimental.list_physical_devices('XLA_GPU'))
        print(tf.config.experimental.list_physical_devices('GPU'))
    print("Devices List: ")
    print(device_lib.list_local_devices())
    if device_type=="GPU":
        device_t="/device:GPU:0"
    else:
        device_t="/device:CPU:0"
        
    if classw==True or classwcomp==True:
        class_weight=class_weight
    else: 
        class_weight=None
    with tf.device(device_t):
        history = model.fit(
            x=[X_train, train_decoder_input], y=y_train, 
            epochs=epochs, 
            validation_data=([X_valid, valid_decoder_input],y_valid),
            callbacks=callbacks,
            batch_size=batch_size,
            class_weight=class_weight,
            sample_weight=sample_weight,
            shuffle=shuffle
        )
        
    # Save model
    model.save("s2s")
    model = keras.models.load_model("s2s")

    encoder_model = Model(encoder_inputs, outputs = [encoder_outputs3, final_enc_h, final_enc_c])
    tf.keras.utils.plot_model(encoder_model, to_file=results_dir+"/model_enc.png", show_shapes=True)
    decoder_state_h = keras.Input(shape=(2*lstm_units,))
    decoder_state_c = keras.Input(shape=(2*lstm_units,))
    decoder_hidden_state_input = keras.Input(shape=(X_train.shape[1],2*lstm_units))
    
    dec_states = [decoder_state_h, decoder_state_c]
    decoder_inputs = model.input[1]  # input_2 
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(decoder_inputs, initial_state=dec_states)
    
    # Attention inference
    attention_result_inf,_ = attention_layer([decoder_hidden_state_input, decoder_outputs2])
    
    decoder_concat_input_inf = layers.Concatenate(axis=-1, name='concat_layer')([decoder_outputs2, attention_result_inf])
    
    dec_states2= [state_h2, state_c2]
    
    decoder_outputs3 = decoder_dense(decoder_concat_input_inf)
    
    decoder_model= Model(
                        [decoder_inputs] + [decoder_hidden_state_input, decoder_state_h, decoder_state_c],
                         [decoder_outputs3]+ dec_states2)
    
    tf.keras.utils.plot_model(decoder_model, to_file=results_dir+"/model_dec.png", show_shapes=True)
    X_valid=encoder_model.predict(X_valid)
    output_tokens, h, c = decoder_model.predict([valid_decoder_input] + X_valid)
    # print(output_tokens)
    labels = (output_tokens >= 0.5).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    print(classification_report(y_valid, y_pred))
    print(confusion_matrix(y_valid,y_pred))
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    
    print("process time: ", ptime, "real time: ", rtime)
    
    # timestr = time.strftime("%Y%m%d_%H%M%S")
    # print("timestr: ", timestr)
    # results_dir="./results"
    # if not os.path.exists(results_dir): os.mkdir(results_dir)
    # timestr='20210508_150602'
    # model = keras.models.load_model(results_dir+'/'+'deeplearning3tW_d1_model_20210508_150602')
    # history_df=pd.read_csv(model_dir+'/'+'deeplearning3tW_d1_model_20210508_150602_history.csv', index_col=0, header=0)
    model_dir=results_dir+'/model_ensb'
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model.save(model_dir+"/model.h5")
    model.save_weights(model_dir+"/weights.h5")
    orig_stdout = sys.stdout
    f = open(model_dir+"/devices.txt", 'w+')
    sys.stdout = f
    for device in device_lib.list_local_devices():
        print(device)
    sys.stdout = orig_stdout
    f.close()
    mparams_file=model_dir+"/modelparams.txt"
    with open(mparams_file, 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.write("program name: "+sys.argv[0]+ '\n')
        f.write("device used: "+device_t+ '\n')
        f.write("optimizer: "+optimizer_name+ '\n')
        f.write("loss: "+loss+ '\n')
        f.write("metrics: "+str(metrics)+ '\n')
        f.write("patience: "+ str(patience)+ '\n')
        f.write("min_delta: "+str(min_delta)+ '\n')
        f.write("restore_best_weights: "+str(restore_best_weights)+ '\n')
        f.write("gru_unit: "+str(gru_unit)+ '\n')
        # f.write("validation_split: "+str(validation_split)+ '\n')
        f.write("batch_size: "+str(batch_size)+ '\n')
        # f.write("steps_per_epoch: "+str(steps_per_epoch)+ '\n')
        f.write("epochs: "+str(epochs)+ '\n')
        # f.write("padding: "+padding+ '\n')
        # f.write("strides: "+str(strides)+ '\n')
        # f.write("kernel_size: "+str(kernel_size)+ '\n')
        # f.write("n_filter: "+str(n_filter)+ '\n')
        # f.write("max_pool: "+str(max_pool)+ '\n')
        f.write("callbacks: "+"[early_stopping]"+ '\n')
        f.write("class_weight: "+str(class_weight)+ '\n')
        f.write("shuffle: "+str(shuffle)+ '\n')
        f.write("Arguments: "+str(vars(args))+ '\n')
        f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ '\n')
        f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ '\n')
        f.write("Test dataset 355 undersampled (1:1): "+str(under_1_1_test_355)+ '\n')
        f.write("Test dataset 448 undersampled (1:1): "+str(under_1_1_test_448)+ '\n')
        f.write("Training dataset: "+training_dataset+ '\n')
        # f.write("The length of training dataset: "+str(len(df1))+ '\n')
        # f.write("The length of case study(cs) dataset: "+str(len(df_cs))+ '\n')
        # f.write("The length of training dataset w/o cs: "+str(len(df))+ '\n')
        f.write("The number of columns of training dataset w/o label: "+str(len(X_train[0][0]))+ '\n')
        f.write("The size of sliding windows : "+str(len(X_train[0]))+ '\n')
        f.write("The length of training dataset after splitting: "+str(len(X_train))+ '\n')
        f.write("The length of validation dataset: "+str(len(X_valid))+ '\n')
        f.close()
        
    plot_model_history(history,model_dir)
    
    metrics_results=[]
    clsf_model='LSTM'
    scaler="Standart"
    scale_data=False
    print("Dtestset 355 \n")
    t_355_decoder_input=np.full((y_355.shape[0],), start_input)
    X_355=encoder_model.predict(X_355)
    test_dataset_355="df_alldataW_Dset_355.csv"
    # output_tokens, h, c = decoder_model.predict([t_355_decoder_input] + X_355)
    # labels = (output_tokens >= 0.5).astype(int)
    # y_pred = np.squeeze(np.asarray(labels))
    # print(classification_report(y_355, y_pred))
    # print(confusion_matrix(y_355,y_pred))
    testset_prediction_eval3(decoder_model,[t_355_decoder_input] + X_355,y_355,[],model_dir,metrics_results,"X_355",scale_data,scaler,clsf_model)
    
    print("Dtestset 448 \n")
    t_448_decoder_input=np.full((y_448.shape[0],), start_input)
    X_448=encoder_model.predict(X_448)
    test_dataset_448="df_alldataW_Dset_448.csv"
    # output_tokens, h, c = decoder_model.predict([t_448_decoder_input] + X_448)
    # print(output_tokens.shape)
    # labels = (output_tokens >= 0.5).astype(int)
    # y_pred = np.squeeze(np.asarray(labels))
    # print(classification_report(y_448, y_pred))
    # print(confusion_matrix(y_448,y_pred))
    testset_prediction_eval3(decoder_model,[t_448_decoder_input] + X_448,y_448,[],model_dir,metrics_results,"X_448",scale_data,scaler,clsf_model)
    # for i in output_tokens:
    #     if i>0.5: print(i,end=' ')
    with open(mparams_file, 'a') as f:
        f.write("Test dataset: "+test_dataset_355+ '\n')
        f.write("The length of test dataset 355: "+str(len(X_355))+ '\n')
        f.write("Test dataset: "+test_dataset_448+ '\n')
        f.write("The length of test dataset 448: "+str(len(X_448))+ '\n')
        f.close()
    df_metrics_all=pd.concat(metrics_results)
    print(df_metrics_all)
    
    df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
    df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")

def GRUs2sAttention(results_dir):
    if undersampling==True:
        under_1_1_training=True
        under_1_1_valid=True
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl="_down"
    else:
        under_1_1_training=False
        under_1_1_valid=False
        under_1_1_test_355=False
        under_1_1_test_448=False
        under_lbl=""
    training_dataset="df_alldataW_all.csv"
    X_train, y_train, X_valid, y_valid, \
    X_355, y_355, \
    X_448, y_448 = load_data(under_lbl)
    
    tf.keras.backend.clear_session()
    optimizer_name='adam'
    learning_rate=0.0005 #0.002
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss='binary_crossentropy'
    metrics=[tf.keras.metrics.AUC(curve='PR')]
    patience=5
    min_delta=1e-5
    restore_best_weights=True
    validation_split=0.2
    batch_size=512
    steps_per_epoch=500
    epochs=15
    padding='same'
    strides=(1, 1) 
    kernel_size=5
    n_filter=48
    max_pool=(2,2)
    gru_units=16
    TIME_STEPS = 2
    INPUT_DIM = 18
    lstm_units = 16
    nb_classes = 1
    num_layr=60
    start_input=-2
    input_shape=X_train.shape[1:]
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights,
        mode='min'
    )
    callbacks=[early_stopping]
    w0=X_train.shape[0]/(2*(X_train.shape[0]-np.sum(y_train,dtype=int)))
    w1=X_train.shape[0]/(2*np.sum(y_train,dtype=int))
    class_weight={0: w0, 1: w1}
    #class_weight={0: 0.55, 1: 4.97}
    if sqrtw==True:
    	w0=math.sqrt(w0)
    	w1=math.sqrt(w1)
    if classwcomp==True: 
        class_weight={0: w0, 1: w1}
    else:
        class_weight={0: 0.55, 1: 4.97}
    print(class_weight)
    if sampw==True:
        sample_weight=np.array([w0 if i==0 else w1 for i in y_train.tolist()])
    else:
        sample_weight=None
    shuffle=True
    print(X_train.shape[0],X_train.shape[1],X_train.shape[2])
    
    start_time = time.time()
    start = time.process_time()
    
    """
    ## Build the model
    """
    
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=X_train.shape[1:])
    
    # Bidirectional lstm layer
    enc_gru1 = layers.Bidirectional(layers.GRU(gru_units,return_sequences=True,return_state=False))
    encoder_outputs1 = enc_gru1(encoder_inputs)
    
    enc_gru2 = layers.Bidirectional(layers.GRU(gru_units,return_sequences=True,return_state=False))
    encoder_outputs2 = enc_gru2(encoder_outputs1)
    
    enc_gru3 = layers.Bidirectional(layers.GRU(gru_units,return_sequences=True,return_state=True))
    encoder_outputs3, forw_state_h, back_state_h = enc_gru3(encoder_outputs2)
    
    final_enc_h = layers.Concatenate()([forw_state_h,back_state_h])
    
    encoder_states =[final_enc_h]
    
    # Set up the decoder, using `encoder_states` as initial state.
    
    decoder_inputs = keras.Input(shape=(None,1))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_gru = layers.GRU(2*gru_units, return_sequences=True, return_state=True)
    decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=encoder_states)
    
    #Attention Layer
    #attention_layer = tf.keras.layers.Attention()
    attention_layer = AttentionLayer()
    attention_result,_= attention_layer([encoder_outputs3,decoder_outputs])
    
    # Concat attention output and decoder LSTM output 
    decoder_concat_input = layers.Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_result])
    
    decoder_dense = keras.layers.Dense(1, activation="sigmoid")
    decoder_outputs = decoder_dense(decoder_concat_input)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    """
    ## Train the model
    """
    train_decoder_input=np.full((y_train.shape[0],), start_input)
    valid_decoder_input=np.full((y_valid.shape[0],), start_input)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    tf.keras.utils.plot_model(model, to_file=results_dir+"/model.png", show_shapes=True)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    if tf.test.is_gpu_available():
        print("GPUs available: ")
        print(tf.config.experimental.list_physical_devices('XLA_GPU'))
        print(tf.config.experimental.list_physical_devices('GPU'))
    print("Devices List: ")
    print(device_lib.list_local_devices())
    if device_type=="GPU":
        device_t="/device:GPU:0"
    else:
        device_t="/device:CPU:0"
        
    if classw==True or classwcomp==True:
        class_weight=class_weight
    else: 
        class_weight=None
    with tf.device(device_t):
        history = model.fit(
            x=[X_train, train_decoder_input], y=y_train, 
            epochs=epochs, 
            validation_data=([X_valid, valid_decoder_input],y_valid),
            callbacks=callbacks,
            batch_size=batch_size,
            class_weight=class_weight,
            sample_weight=sample_weight,
            shuffle=shuffle
        )
        
    # Save model
    model.save("s2s")
    model = keras.models.load_model("s2s")

    encoder_model = Model(encoder_inputs, outputs = [encoder_outputs3, final_enc_h])
    tf.keras.utils.plot_model(encoder_model, to_file=results_dir+"/model_enc.png", show_shapes=True)
    decoder_state_h = keras.Input(shape=(2*gru_units,))
    decoder_hidden_state_input = keras.Input(shape=(X_train.shape[1],2*gru_units))
    
    dec_states = [decoder_state_h]
    decoder_inputs = model.input[1]  # input_2
    decoder_outputs2, state_h2 = decoder_gru(decoder_inputs, initial_state=dec_states)
    
    # Attention inference
    attention_result_inf,_ = attention_layer([decoder_hidden_state_input, decoder_outputs2])
    
    decoder_concat_input_inf = layers.Concatenate(axis=-1, name='concat_layer')([decoder_outputs2, attention_result_inf])
    
    dec_states2= [state_h2]
    
    decoder_outputs3 = decoder_dense(decoder_concat_input_inf)
    
    decoder_model= Model(
                        [decoder_inputs] + [decoder_hidden_state_input, decoder_state_h],
                         [decoder_outputs3]+ dec_states2)
    
    tf.keras.utils.plot_model(decoder_model, to_file=results_dir+"/model_dec.png", show_shapes=True)
    X_valid=encoder_model.predict(X_valid)
    output_tokens, h = decoder_model.predict([valid_decoder_input] + X_valid)
    # print(output_tokens)
    labels = (output_tokens >= 0.5).astype(int)
    y_pred = np.squeeze(np.asarray(labels))
    print(classification_report(y_valid, y_pred))
    print(confusion_matrix(y_valid,y_pred))
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    
    print("process time: ", ptime, "real time: ", rtime)
    
    # timestr = time.strftime("%Y%m%d_%H%M%S")
    # print("timestr: ", timestr)
    # results_dir="./results"
    # if not os.path.exists(results_dir): os.mkdir(results_dir)
    # timestr='20210508_150602'
    # model = keras.models.load_model(results_dir+'/'+'deeplearning3tW_d1_model_20210508_150602')
    # history_df=pd.read_csv(model_dir+'/'+'deeplearning3tW_d1_model_20210508_150602_history.csv', index_col=0, header=0)
    model_dir=results_dir+'/model_ensb'
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model.save(model_dir+"/model.h5")
    model.save_weights(model_dir+"/weights.h5")
    orig_stdout = sys.stdout
    f = open(model_dir+"/devices.txt", 'w+')
    sys.stdout = f
    for device in device_lib.list_local_devices():
        print(device)
    sys.stdout = orig_stdout
    f.close()
    mparams_file=model_dir+"/modelparams.txt"
    with open(mparams_file, 'w+') as f:
        with redirect_stdout(f):
            model.summary()
        f.write("program name: "+sys.argv[0]+ '\n')
        f.write("device used: "+device_t+ '\n')
        f.write("optimizer: "+optimizer_name+ '\n')
        f.write("loss: "+loss+ '\n')
        f.write("metrics: "+str(metrics)+ '\n')
        f.write("patience: "+ str(patience)+ '\n')
        f.write("min_delta: "+str(min_delta)+ '\n')
        f.write("restore_best_weights: "+str(restore_best_weights)+ '\n')
        f.write("gru_unit: "+str(gru_units)+ '\n')
        # f.write("validation_split: "+str(validation_split)+ '\n')
        f.write("batch_size: "+str(batch_size)+ '\n')
        # f.write("steps_per_epoch: "+str(steps_per_epoch)+ '\n')
        f.write("epochs: "+str(epochs)+ '\n')
        # f.write("padding: "+padding+ '\n')
        # f.write("strides: "+str(strides)+ '\n')
        # f.write("kernel_size: "+str(kernel_size)+ '\n')
        # f.write("n_filter: "+str(n_filter)+ '\n')
        # f.write("max_pool: "+str(max_pool)+ '\n')
        f.write("callbacks: "+"[early_stopping]"+ '\n')
        f.write("class_weight: "+str(class_weight)+ '\n')
        f.write("shuffle: "+str(shuffle)+ '\n')
        f.write("Arguments: "+str(vars(args))+ '\n')
        f.write("Training dataset undersampled (1:1): "+str(under_1_1_training)+ '\n')
        f.write("Training(valid) dataset undersampled (1:1): "+str(under_1_1_valid)+ '\n')
        f.write("Test dataset 355 undersampled (1:1): "+str(under_1_1_test_355)+ '\n')
        f.write("Test dataset 448 undersampled (1:1): "+str(under_1_1_test_448)+ '\n')
        f.write("Training dataset: "+training_dataset+ '\n')
        # f.write("The length of training dataset: "+str(len(df1))+ '\n')
        # f.write("The length of case study(cs) dataset: "+str(len(df_cs))+ '\n')
        # f.write("The length of training dataset w/o cs: "+str(len(df))+ '\n')
        f.write("The number of columns of training dataset w/o label: "+str(len(X_train[0][0]))+ '\n')
        f.write("The size of sliding windows : "+str(len(X_train[0]))+ '\n')
        f.write("The length of training dataset after splitting: "+str(len(X_train))+ '\n')
        f.write("The length of validation dataset: "+str(len(X_valid))+ '\n')
        f.close()
        
    plot_model_history(history,model_dir)
    
    metrics_results=[]
    clsf_model='LSTM'
    scaler="Standart"
    scale_data=False
    print("Dtestset 355 \n")
    t_355_decoder_input=np.full((y_355.shape[0],), start_input)
    X_355=encoder_model.predict(X_355)
    test_dataset_355="df_alldataW_Dset_355.csv"
    # output_tokens, h, c = decoder_model.predict([t_355_decoder_input] + X_355)
    # labels = (output_tokens >= 0.5).astype(int)
    # y_pred = np.squeeze(np.asarray(labels))
    # print(classification_report(y_355, y_pred))
    # print(confusion_matrix(y_355,y_pred))
    testset_prediction_eval4(decoder_model,[t_355_decoder_input] + X_355,y_355,[],model_dir,metrics_results,"X_355",scale_data,scaler,clsf_model)
    
    print("Dtestset 448 \n")
    t_448_decoder_input=np.full((y_448.shape[0],), start_input)
    X_448=encoder_model.predict(X_448)
    test_dataset_448="df_alldataW_Dset_448.csv"
    # output_tokens, h, c = decoder_model.predict([t_448_decoder_input] + X_448)
    # print(output_tokens.shape)
    # labels = (output_tokens >= 0.5).astype(int)
    # y_pred = np.squeeze(np.asarray(labels))
    # print(classification_report(y_448, y_pred))
    # print(confusion_matrix(y_448,y_pred))
    testset_prediction_eval4(decoder_model,[t_448_decoder_input] + X_448,y_448,[],model_dir,metrics_results,"X_448",scale_data,scaler,clsf_model)
    # for i in output_tokens:
    #     if i>0.5: print(i,end=' ')
    with open(mparams_file, 'a') as f:
        f.write("Test dataset: "+test_dataset_355+ '\n')
        f.write("The length of test dataset 355: "+str(len(X_355))+ '\n')
        f.write("Test dataset: "+test_dataset_448+ '\n')
        f.write("The length of test dataset 448: "+str(len(X_448))+ '\n')
        f.close()
    df_metrics_all=pd.concat(metrics_results)
    print(df_metrics_all)
    
    df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
    df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")

def kfoldCatboost(results_dir,n_folds,wsize=""):
    under_1_1_training, under_1_1_valid, under_lbl, training_dataset = data_info(undersampling,all_data)
    Xy_datasets = load_dataML(under_lbl,wsize)
    X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
    results_dir = create_result_folder2(results_dir,wsize,X_train)
    
    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold , auc_per_fold , loss_per_fold = [], [], []
    precision_per_fold, recall_per_fold, aupr_per_fold=[], [], []
    f1_per_fold, specivity_per_fold, mcc_per_fold=[], [], []
    mse_per_fold, mn_mtr_ls=[], []
    scaler = StandardScaler()
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        if int(fld_no)!=-1 and fold_no!=int(fld_no): 
            fold_no+=1
            continue
        if all_data==True:
            print(y_train.shape)
            train, validation = X_train.iloc[train_index,:], X_train.iloc[validation_index,:]
            target_train, target_val = y_train.iloc[train_index], y_train.iloc[validation_index]
        else:
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid
            
        if scale_data==True:
            tests=[validation]+Xy_datasets[4::2] if pdbsets==True else [validation]+Xy_datasets[10::2]
            train, tests, scaler = scale_datasets2(train,tests,scaler_type)

        print("\nFold ",fold_no)
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        
        ### PARAMETERS ###
        # metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000,name='auprc'),
        #          tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000,name='auroc'),
        #          'binary_accuracy','mse','Precision','Recall',
        #          'TrueNegatives','FalsePositives',
        #          'FalseNegatives','TruePositives']
        # metrics=['Precision','Recall','AUC']
        patience=4
        min_delta=1e-5
        restore_best_weights=True
        batch_size=1024
        epochs=9
        clsf_model="Catboost"
        tag=clsf_model.lower()
        device_type="CPU"
        prog_name=sys.argv[0]
        shuffle=True
        monitor,mode,metric_name="val_auprc",'max',"AUPRC"
        class_num=np.unique(target_train)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=restore_best_weights,
            mode=mode
        )
        callbacks=[early_stopping]
        classWeight = classWeight_comp(class_num,target_train,sqrtw,classwcomp)
        print(classWeight)
        sample_weight = sampleWeight_comp(classWeight,target_train,sampw)
        ### =========== ###
        
        ### MODEL ###
        model= CatBoostClassifier(class_weights=classWeight,
                                   loss_function='Logloss',eval_metric="PRAUC",verbose=0)

        # Define checkpoint callback
        model_dir=results_dir+'/model_catboost_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
            early_stopping
            # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True, mode='max')
        ]
        
        device_t="/device:"+device_type_control(model_dir,device_type)+":0"
        
        classWeight = class_wght(classw,classwcomp,classWeight)
        print(classWeight)
        
        validation_data, callbacks = vald_data_callback(all_data,novald,
                                                        validation,target_val,callbacks_check)
        ### FIT MODEL ###
        with tf.device(device_t):
            model.fit(train, target_train,
                      eval_set=validation_data,
                      sample_weight=sample_weight)
            # history = model.fit(
            #     train, target_train, 
            #     epochs=epochs, 
            #     validation_data=validation_data,
            #     callbacks=callbacks,
            #     batch_size=batch_size,
            #     class_weight=classWeight,
            #     sample_weight=sample_weight,
            #     shuffle=shuffle
            # )
        
        # Save model
        # model.save(model_dir+'/model.tf')
        # model.save_weights(model_dir+"/weights.tf")
        filename=model_dir+"/"+'model.pkl'
        joblib.dump(model, filename)
        model.save_model(filename[:-3]+'bin')
        probas = model.predict_proba(validation)
        probas = np.delete(probas, 0, 1) #delete first column
        plot_probas_valid(probas,model_dir)
        # plot_distrb_pred_valid(probas,target_val,model_dir)
        labels = (probas >= 0.5).astype(int)
        y_pred = np.squeeze(np.asarray(labels))
        # y_pred = model.predict(validation)
        print(classification_report(target_val, y_pred))
        print(confusion_matrix(target_val,y_pred))
        
        # scores = model.evaluate(validation, target_val, verbose=0)
        # print_validation_results_fold(model, scores, fold_no)
        # scores=model.eval_metrics(Pool(validation, target_val),metrics)
        # print(scores)
        # print(model.metrics_names)
        
        # specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc = compute_metrics(model,scores)
        logit = lambda x: log(x / (1 - x))
        approxes = list(map(logit, probas))
        print('\nScoring metrics for validation set for fold',fold_no)
        recall_scr = eval_metric(target_val, approxes, 'Recall')[0]
        print('Recall:', recall_scr)
        precision_scr = eval_metric(target_val, approxes, 'Precision')[0]
        print('Precision:',precision_scr)
        accuracy_scr = eval_metric(target_val, approxes, 'Accuracy')[0]
        print('Accuracy:',accuracy_scr)
        f_1_scr = eval_metric(target_val, approxes, 'F1')[0]
        print('F1:', f_1_scr)
        mcc_score = eval_metric(target_val, approxes, 'MCC')[0]
        print('MCC:', mcc_score)
        auc_scr = eval_metric(target_val, approxes, 'AUC')[0]
        print('AUROC:',auc_scr)
        aupr_scr = eval_metric(target_val, approxes, 'PRAUC')[0]
        print('AUPRC:',aupr_scr)
        print('\n')
        # loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold, \
        # precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold, \
        # auc_per_fold,aupr_per_fold = \
        # metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
        #                        loss_per_fold,mse_per_fold,recall_per_fold,
        #                        specivity_per_fold,precision_per_fold,acc_per_fold,
        #                        f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold)
        fold_no = fold_no + 1
        
        # print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold)
        
        # print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
                                           # precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
                                           # auc_per_fold,aupr_per_fold)
        
        mparams_file=model_dir+"/modelparams.txt"
        
        write_model_parameters_ML(mparams_file,model,prog_name,device_t,
                                   classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                                   training_dataset,train,validation)
        
        # plot_model_history(history,model_dir)
        del train,validation,target_train,target_val
        gc.collect()
        metrics_results=[]
                
        test_tags,tsets_thrshlds_dict=test_tags_thresholds(pdbsets)
        X_testsets=select_testsets(pdbsets,Xy_datasets)
        test_results_ML2(model,X_testsets,tsets_thrshlds_dict,model_dir,
                             metrics_results,scale_data,scaler,clsf_model)
        
        append_model_parameters_testsets(mparams_file, X_testsets)
        append_thresholds_testsets(mparams_file, tsets_thrshlds_dict)
        
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        
        test_mtr=[i+"_1" for i in test_tags]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')[metric_name].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        del model
        gc.collect()
        if all_data==False and novald==False and int(fld_no)!=-1: break
        # if fold_no==3: break
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print_results_folds(metric_name,mn_mtr_ls)
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/'+tag+'.dir', 'w') as f:
        f.write(results_dir+f'/model_{tag}_fold/fold_{fd_slct}/model.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)
    
def kfoldLightgbm(results_dir,n_folds,wsize=""):
    under_1_1_training, under_1_1_valid, under_lbl, training_dataset = data_info(undersampling,all_data)
    Xy_datasets = load_dataML(under_lbl,wsize)
    X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
    results_dir = create_result_folder2(results_dir,wsize,X_train)
    
    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold , auc_per_fold , loss_per_fold = [], [], []
    precision_per_fold, recall_per_fold, aupr_per_fold=[], [], []
    f1_per_fold, specivity_per_fold, mcc_per_fold=[], [], []
    mse_per_fold, mn_mtr_ls=[], []
    scaler = StandardScaler()
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        if int(fld_no)!=-1 and fold_no!=int(fld_no): 
            fold_no+=1
            continue
        if all_data==True:
            print(y_train.shape)
            train, validation = X_train.iloc[train_index,:], X_train.iloc[validation_index,:]
            target_train, target_val = y_train.iloc[train_index], y_train.iloc[validation_index]
        else:
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid
            
        if scale_data==True:
            tests=[validation]+Xy_datasets[4::2] if pdbsets==True else [validation]+Xy_datasets[10::2]
            train, tests, scaler = scale_datasets2(train,tests,scaler_type)

        print("\nFold ",fold_no)
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        
        ### PARAMETERS ###
        # metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000,name='auprc'),
        #          tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000,name='auroc'),
        #          'binary_accuracy','mse','Precision','Recall',
        #          'TrueNegatives','FalsePositives',
        #          'FalseNegatives','TruePositives']
        # metrics=['Precision','Recall','AUC']
        patience=10
        min_delta=1e-5
        restore_best_weights=True
        batch_size=1024
        epochs=9
        clsf_model="Lightgbm"
        tag=clsf_model.lower()
        device_type="CPU"
        prog_name=sys.argv[0]
        shuffle=True
        monitor,mode,metric_name="val_auprc",'max',"AUPRC"
        class_num=np.unique(target_train)
        early_stopping = lgb.early_stopping(
            stopping_rounds=patience,
            first_metric_only =True
            # min_delta=min_delta
        )
        callbacks=[early_stopping]
        classWeight = classWeight_comp(class_num,target_train,sqrtw,classwcomp)
        print(classWeight)
        sample_weight = sampleWeight_comp(classWeight,target_train,sampw)
        ### =========== ###
        
        ### MODEL ###
        model = lgb.LGBMClassifier(class_weight=classWeight) 

        # Define checkpoint callback
        model_dir=results_dir+'/model_lightgbm_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
            early_stopping
            # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True, mode='max')
        ]
        
        device_t="/device:"+device_type_control(model_dir,device_type)+":0"
        
        classWeight = class_wght(classw,classwcomp,classWeight)
        print(classWeight)
        
        validation_data, callbacks = vald_data_callback(all_data,novald,
                                                        validation,target_val,callbacks_check)
        ### FIT MODEL ###
        with tf.device(device_t):
            model.fit(train, target_train,
                      eval_set=validation_data,
                      sample_weight=sample_weight,
                      callbacks=callbacks,
                      eval_metric = ['average_precision'])
            # history = model.fit(
            #     train, target_train, 
            #     epochs=epochs, 
            #     validation_data=validation_data,
            #     callbacks=callbacks,
            #     batch_size=batch_size,
            #     class_weight=classWeight,
            #     sample_weight=sample_weight,
            #     shuffle=shuffle
            # )
        
        # Save model
        # model.save(model_dir+'/model.tf')
        # model.save_weights(model_dir+"/weights.tf")
        filename=model_dir+"/"+'model.pkl'
        joblib.dump(model, filename)
        model.booster_.save_model(filename[:-3]+'bin')
        probas = model.predict_proba(validation)
        probas = np.delete(probas, 0, 1) #delete first column
        plot_probas_valid(probas,model_dir)
        # plot_distrb_pred_valid(probas,target_val,model_dir)
        labels = (probas >= 0.5).astype(int)
        y_pred = np.squeeze(np.asarray(labels))
        # y_pred = model.predict(validation)
        print(classification_report(target_val, y_pred))
        print(confusion_matrix(target_val,y_pred))
        
        # scores = model.evaluate(validation, target_val, verbose=0)
        # print_validation_results_fold(model, scores, fold_no)
        # scores=model.eval_metrics(Pool(validation, target_val),metrics)
        # print(scores)
        # print(model.metrics_names)
        
        # specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc = compute_metrics(model,scores)
        try:
            logit = lambda x: log(x / (1 - x))
            approxes = list(map(logit, probas))
            print('\nScoring metrics for validation set for fold',fold_no)
            recall_scr = eval_metric(target_val, approxes, 'Recall')[0]
            print('Recall:', recall_scr)
            precision_scr = eval_metric(target_val, approxes, 'Precision')[0]
            print('Precision:',precision_scr)
            accuracy_scr = eval_metric(target_val, approxes, 'Accuracy')[0]
            print('Accuracy:',accuracy_scr)
            f_1_scr = eval_metric(target_val, approxes, 'F1')[0]
            print('F1:', f_1_scr)
            mcc_score = eval_metric(target_val, approxes, 'MCC')[0]
            print('MCC:', mcc_score)
            auc_scr = eval_metric(target_val, approxes, 'AUC')[0]
            print('AUROC:',auc_scr)
            aupr_scr = eval_metric(target_val, approxes, 'PRAUC')[0]
            print('AUPRC:',aupr_scr)
            print('\n')
        except:
            print('\nScoring metrics for validation set for fold',fold_no)
            print('There is a divison by zero \n')
        # loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold, \
        # precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold, \
        # auc_per_fold,aupr_per_fold = \
        # metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
        #                        loss_per_fold,mse_per_fold,recall_per_fold,
        #                        specivity_per_fold,precision_per_fold,acc_per_fold,
        #                        f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold)
        fold_no = fold_no + 1
        
        # print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold)
        
        # print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
                                           # precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
                                           # auc_per_fold,aupr_per_fold)
        
        mparams_file=model_dir+"/modelparams.txt"
        
        write_model_parameters_ML_lgbm(mparams_file,model,prog_name,device_t,
                                   classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                                   training_dataset,train,validation)
        
        # plot_model_history(history,model_dir)
        del train,validation,target_train,target_val
        gc.collect()
        metrics_results=[]
                
        test_tags,tsets_thrshlds_dict=test_tags_thresholds(pdbsets)
        X_testsets=select_testsets(pdbsets,Xy_datasets)
        test_results_ML2(model,X_testsets,tsets_thrshlds_dict,model_dir,
                             metrics_results,scale_data,scaler,clsf_model)
        
        append_model_parameters_testsets(mparams_file, X_testsets)
        append_thresholds_testsets(mparams_file, tsets_thrshlds_dict)
        
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        
        test_mtr=[i+"_1" for i in test_tags]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')[metric_name].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        del model
        gc.collect()
        if all_data==False and novald==False and int(fld_no)!=-1: break
        # if fold_no==3: break
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print_results_folds(metric_name,mn_mtr_ls)
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/'+tag+'.dir', 'w') as f:
        f.write(results_dir+f'/model_{tag}_fold/fold_{fd_slct}/model.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)

def kfoldXgboost(results_dir,n_folds,wsize=""):
    under_1_1_training, under_1_1_valid, under_lbl, training_dataset = data_info(undersampling,all_data)
    Xy_datasets = load_dataML(under_lbl,wsize)
    X_train, y_train, X_valid, y_valid = Xy_datasets[0], Xy_datasets[1], Xy_datasets[2], Xy_datasets[3]
    results_dir = create_result_folder2(results_dir,wsize,X_train)
    
    start_time = time.time()
    start = time.process_time()
    
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    # score containers
    acc_per_fold , auc_per_fold , loss_per_fold = [], [], []
    precision_per_fold, recall_per_fold, aupr_per_fold=[], [], []
    f1_per_fold, specivity_per_fold, mcc_per_fold=[], [], []
    mse_per_fold, mn_mtr_ls=[], []
    scaler = StandardScaler()
    # K-fold Cross Validation
    fold_no=1
    for train_index, validation_index in kf.split(X_train, y_train):
        if int(fld_no)!=-1 and fold_no!=int(fld_no): 
            fold_no+=1
            continue
        if all_data==True:
            print(y_train.shape)
            train, validation = X_train.iloc[train_index,:], X_train.iloc[validation_index,:]
            target_train, target_val = y_train.iloc[train_index], y_train.iloc[validation_index]
        else:
            train, validation = X_train, X_valid
            target_train, target_val = y_train, y_valid
            
        if scale_data==True:
            tests=[validation]+Xy_datasets[4::2] if pdbsets==True else [validation]+Xy_datasets[10::2]
            train, tests, scaler = scale_datasets2(train,tests,scaler_type)

        print("\nFold ",fold_no)
        print("Input shape for training data: ",train.shape)
        print("Input shape for validation data: ",validation.shape)
        print("Input shape for model: ",train.shape[1:])
        
        ### PARAMETERS ###
        # metrics=[tf.keras.metrics.AUC(curve='PR',num_thresholds=1000,name='auprc'),
        #          tf.keras.metrics.AUC(curve='ROC',num_thresholds=1000,name='auroc'),
        #          'binary_accuracy','mse','Precision','Recall',
        #          'TrueNegatives','FalsePositives',
        #          'FalseNegatives','TruePositives']
        # metrics=['Precision','Recall','AUC']
        patience=4
        min_delta=1e-5
        restore_best_weights=True
        batch_size=1024
        epochs=9
        clsf_model="Xgboost"
        tag=clsf_model.lower()
        device_type="CPU"
        prog_name=sys.argv[0]
        shuffle=True
        monitor,mode,metric_name="val_auprc",'max',"AUPRC"
        class_num=np.unique(target_train)
        early_stopping = xgb.callback.EarlyStopping(rounds=patience,
                                        save_best=True)
        callbacks=[early_stopping]
        classWeight = classWeight_comp(class_num,target_train,sqrtw,classwcomp)
        print(classWeight)
        sample_weight = sampleWeight_comp(classWeight,target_train,sampw)
        sample_weight_val = sampleWeight_comp(classWeight,target_val,sampw)
        ### =========== ###
        
        ### MODEL ###
        model = xgb.XGBClassifier(use_label_encoder=False,n_jobs=20) 
        
        # Define checkpoint callback
        model_dir=results_dir+'/model_xgboost_fold'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        model_dir=model_dir+f'/fold_{fold_no}'
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        checkpoint_path = model_dir+f'/fold_{fold_no}.tf'
        callbacks_check = [
            early_stopping
            # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auroc', save_best_only=True, mode='max')
        ]
        
        device_t="/device:"+device_type_control(model_dir,device_type)+":0"
        
        classWeight = class_wght(classw,classwcomp,classWeight)
        print(classWeight)
        
        validation_data, callbacks = vald_data_callback(all_data,novald,
                                                        validation,target_val,callbacks_check)
        ### FIT MODEL ###
        with tf.device(device_t):
            model.fit(train, target_train,
                      eval_set=[(train, target_train),validation_data],
                      sample_weight=sample_weight,
                      sample_weight_eval_set=[sample_weight,sample_weight_val],
                      callbacks=callbacks,
                      eval_metric = ['aucpr'])
            # history = model.fit(
            #     train, target_train, 
            #     epochs=epochs, 
            #     validation_data=validation_data,
            #     callbacks=callbacks,
            #     batch_size=batch_size,
            #     class_weight=classWeight,
            #     sample_weight=sample_weight,
            #     shuffle=shuffle
            # )

        # Save model
        # model.save(model_dir+'/model.tf')
        # model.save_weights(model_dir+"/weights.tf")
        filename=model_dir+"/"+'model.pkl'
        joblib.dump(model, filename)
        model.save_model(filename[:-3]+'bin')
        probas = model.predict_proba(validation)
        probas = np.delete(probas, 0, 1) #delete first column
        plot_probas_valid(probas,model_dir)
        # plot_distrb_pred_valid(probas,target_val,model_dir)
        labels = (probas >= 0.5).astype(int)
        y_pred = np.squeeze(np.asarray(labels))
        # y_pred = model.predict(validation)
        print(classification_report(target_val, y_pred))
        print(confusion_matrix(target_val,y_pred))
        
        # scores = model.evaluate(validation, target_val, verbose=0)
        # print_validation_results_fold(model, scores, fold_no)
        # scores=model.eval_metrics(Pool(validation, target_val),metrics)
        # print(scores)
        # print(model.metrics_names)
        
        # specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc = compute_metrics(model,scores)
        try:
            logit = lambda x: log(x / (1 - x))
            approxes = list(map(logit, probas))
            print('\nScoring metrics for validation set for fold',fold_no)
            recall_scr = eval_metric(target_val, approxes, 'Recall')[0]
            print('Recall:', recall_scr)
            precision_scr = eval_metric(target_val, approxes, 'Precision')[0]
            print('Precision:',precision_scr)
            accuracy_scr = eval_metric(target_val, approxes, 'Accuracy')[0]
            print('Accuracy:',accuracy_scr)
            f_1_scr = eval_metric(target_val, approxes, 'F1')[0]
            print('F1:', f_1_scr)
            mcc_score = eval_metric(target_val, approxes, 'MCC')[0]
            print('MCC:', mcc_score)
            auc_scr = eval_metric(target_val, approxes, 'AUC')[0]
            print('AUROC:',auc_scr)
            aupr_scr = eval_metric(target_val, approxes, 'PRAUC')[0]
            print('AUPRC:',aupr_scr)
            print('\n')
        except:
            print('\nScoring metrics for validation set for fold',fold_no)
            print('There is a divison by zero \n')
        # loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold, \
        # precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold, \
        # auc_per_fold,aupr_per_fold = \
        # metrics_containers(model, scores, specivity_sc, recall_sc, precision_sc, f1_sc, mcc_sc,
        #                        loss_per_fold,mse_per_fold,recall_per_fold,
        #                        specivity_per_fold,precision_per_fold,acc_per_fold,
        #                        f1_per_fold,mcc_per_fold,auc_per_fold,aupr_per_fold)
        fold_no = fold_no + 1
        
        # print_scores_per_fold(loss_per_fold,acc_per_fold,precision_per_fold,recall_per_fold)
        
        # print_average_scores_all_folds(loss_per_fold,mse_per_fold,recall_per_fold,specivity_per_fold,
                                           # precision_per_fold,acc_per_fold,f1_per_fold,mcc_per_fold,
                                           # auc_per_fold,aupr_per_fold)
        
        mparams_file=model_dir+"/modelparams.txt"
        
        write_model_parameters_ML_xgb(mparams_file,model,prog_name,device_t,
                                   classWeight,shuffle,args,scale_data,scaler,under_1_1_training, under_1_1_valid, 
                                   training_dataset,train,validation)
        
        # plot_model_history(history,model_dir)
        del train,validation,target_train,target_val
        gc.collect()
        metrics_results=[]
                
        test_tags,tsets_thrshlds_dict=test_tags_thresholds(pdbsets)
        X_testsets=select_testsets(pdbsets,Xy_datasets)
        test_results_ML2(model,X_testsets,tsets_thrshlds_dict,model_dir,
                             metrics_results,scale_data,scaler,clsf_model)
        
        append_model_parameters_testsets(mparams_file, X_testsets)
        append_thresholds_testsets(mparams_file, tsets_thrshlds_dict)
        
        df_metrics_all=pd.concat(metrics_results)
        print(df_metrics_all)
        
        test_mtr=[i+"_1" for i in test_tags]
        mn_mtr=df_metrics_all.query(f'index in {test_mtr}')[metric_name].mean(axis=0)
        # print(mn_mtr)
        mn_mtr_ls.append(mn_mtr)
        df_metrics_all.to_csv(model_dir+"/metrics_results.csv")
        df_metrics_all.to_excel(model_dir+"/metrics_results.xlsx")
        del model
        gc.collect()
        if all_data==False and novald==False and int(fld_no)!=-1: break
        # if fold_no==3: break
    end = time.process_time()
    end_time = time.time()
    ptime = end-start
    rtime=end_time-start_time
    print_results_folds(metric_name,mn_mtr_ls)
    fd_slct=mn_mtr_ls.index(max(mn_mtr_ls))+1
    with open(results_dir+'/'+tag+'.dir', 'w') as f:
        f.write(results_dir+f'/model_{tag}_fold/fold_{fd_slct}/model.tf\n')
        f.close()
    print("process time: ", ptime, "real time: ", rtime)

def main():
    main_results_dir="./results"
    if not os.path.exists(main_results_dir):
        try:
            os.mkdir(main_results_dir)
        except:
            pass
    timestr = time.strftime("%Y%m%d_%H%M%S")
    # timestr = "20211230_223054"
    print("timestr: ", timestr)
    results_dir=main_results_dir+"/results_"+timestr
    # RNN(results_dir)
    # CNN(results_dir)
    # Ensemble(results_dir)
    # AttentionGRU(results_dir)
    # tuningRNN(results_dir)
    # LSTMs2sAttention(results_dir)
    # GRUs2sAttention(results_dir)
    # kfoldLSTMs2s2(results_dir,5)
    # kfoldLSTMs2sAttention2(results_dir,5)
    # kfoldCatboost(results_dir,5,wsize)
    # kfoldLightgbm(results_dir,5,wsize)
    # kfoldXgboost(results_dir,5,wsize)
    mdl_no=int(model_no)
    if mdl_no==1: kfoldRNN(results_dir,5)
    if mdl_no==2: kfoldCNN(results_dir,5)
    if mdl_no==3: kfoldGRUs2s(results_dir,5)
    if mdl_no==4: kfoldGRUs2sAttention(results_dir,5)
    if mdl_no==5: kfoldEnsemble4(results_dir,5)
if __name__ == "__main__":
    main()
