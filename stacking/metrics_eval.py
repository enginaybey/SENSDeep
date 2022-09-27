# -*- coding: utf-8 -*-
"""
@author: engin aybey
"""

import pandas as pd
import matplotlib.pyplot as plt
import os,gc
from sklearn.metrics import classification_report, matthews_corrcoef, balanced_accuracy_score
from sklearn.metrics import roc_auc_score,plot_roc_curve,roc_curve,auc,precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve, plot_confusion_matrix
from sklearn.metrics import average_precision_score,confusion_matrix,ConfusionMatrixDisplay,RocCurveDisplay

def metric_results_macro(clsf,X,y,y_pred,dataset,y_probas=[]):
    df_metrics=pd.DataFrame(columns=['Sensitivity','Specificity','Precision','Accuracy','F1','MCC','AUROC'])
    mcc=matthews_corrcoef(y, y_pred)
    balanced_acc=balanced_accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    macro_precision =  report['macro avg']['precision'] 
    recall_macro = report['macro avg']['recall']   
    f1_macro = report['macro avg']['f1-score']
    accuracy = report['accuracy']
    if len(y_probas)==0: y_probas = clsf.predict_proba(X)[:,1]
    auroc=roc_auc_score(y, y_probas, average=None)
    average_precision = average_precision_score(y, y_probas, average=None)
    df_metrics.loc[dataset,'MCC']=mcc
    df_metrics.loc[dataset,'AUROC']=auroc
    df_metrics.loc[dataset,'Precision']=macro_precision
    df_metrics.loc[dataset,'Sensitivity']=recall_macro
    df_metrics.loc[dataset,'Specificity']=None
    df_metrics.loc[dataset,'F1']=f1_macro
    df_metrics.loc[dataset,'Accuracy']=accuracy
    df_metrics.loc[dataset,'AUPRC']=average_precision
    df_metrics.loc[dataset,'Balanced Accuracy']=balanced_acc
    del y_probas
    gc.collect()
    return df_metrics
def metric_results_weighted(clsf,X,y,y_pred,dataset,y_probas=''):
    df_metrics=pd.DataFrame(columns=['Sensitivity','Specificity','Precision','Accuracy','F1','MCC','AUROC'])
    mcc=matthews_corrcoef(y, y_pred)
    balanced_acc=balanced_accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    macro_precision =  report['weighted avg']['precision'] 
    recall_weighted = report['weighted avg']['recall']   
    f1_weighted = report['weighted avg']['f1-score']
    accuracy = report['accuracy']
    if len(y_probas)==0: y_probas = clsf.predict_proba(X)[:,1]
    auroc=roc_auc_score(y, y_probas, average=None)
    average_precision = average_precision_score(y, y_probas, average=None)
    df_metrics.loc[dataset,'MCC']=mcc
    df_metrics.loc[dataset,'AUROC']=auroc
    df_metrics.loc[dataset,'Precision']=macro_precision
    df_metrics.loc[dataset,'Sensitivity']=recall_weighted
    df_metrics.loc[dataset,'Specificity']=None
    df_metrics.loc[dataset,'F1']=f1_weighted
    df_metrics.loc[dataset,'Accuracy']=accuracy
    df_metrics.loc[dataset,'AUPRC']=average_precision
    df_metrics.loc[dataset,'Balanced Accuracy']=balanced_acc
    del y_probas
    gc.collect()
    return df_metrics
def metric_results_0(clsf,X,y,y_pred,dataset,y_probas=''):
    df_metrics=pd.DataFrame(columns=['Sensitivity','Specificity','Precision','Accuracy','F1','MCC','AUROC'])
    mcc=matthews_corrcoef(y, y_pred)
    balanced_acc=balanced_accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    precision_0 =  report['0']['precision'] 
    recall_0 = report['0']['recall']   
    recall_1 = report['1']['recall'] 
    f1_0 = report['0']['f1-score']
    accuracy = report['accuracy']
    if len(y_probas)==0: y_probas = clsf.predict_proba(X)[:,1]
    auroc=roc_auc_score(y, y_probas, average=None)
    average_precision = average_precision_score(y, y_probas, average=None)
    df_metrics.loc[dataset,'MCC']=mcc
    df_metrics.loc[dataset,'AUROC']=auroc
    df_metrics.loc[dataset,'Precision']=precision_0
    df_metrics.loc[dataset,'Sensitivity']=recall_0
    df_metrics.loc[dataset,'Specificity']=recall_1
    df_metrics.loc[dataset,'F1']=f1_0
    df_metrics.loc[dataset,'Accuracy']=accuracy
    df_metrics.loc[dataset,'AUPRC']=average_precision
    df_metrics.loc[dataset,'Balanced Accuracy']=balanced_acc
    del y_probas
    gc.collect()
    return df_metrics
def metric_results_1(clsf,X,y,y_pred,dataset,y_probas=''):
    df_metrics=pd.DataFrame(columns=['Sensitivity','Specificity','Precision','Accuracy','F1','MCC','AUROC'])
    mcc=matthews_corrcoef(y, y_pred)
    balanced_acc=balanced_accuracy_score(y, y_pred)
    auroc=roc_auc_score(y, y_pred, average=None)
    report = classification_report(y, y_pred, output_dict=True)
    precision_1 =  report['1']['precision'] 
    recall_0 = report['0']['recall']   
    recall_1 = report['1']['recall'] 
    f1_1 = report['1']['f1-score']
    accuracy = report['accuracy']
    if len(y_probas)==0: y_probas = clsf.predict_proba(X)[:,1]
    auroc=roc_auc_score(y, y_probas, average=None)
    average_precision = average_precision_score(y, y_probas, average=None)
    df_metrics.loc[dataset,'MCC']=mcc
    df_metrics.loc[dataset,'AUROC']=auroc
    df_metrics.loc[dataset,'Precision']=precision_1
    df_metrics.loc[dataset,'Sensitivity']=recall_1
    df_metrics.loc[dataset,'Specificity']=recall_0
    df_metrics.loc[dataset,'F1']=f1_1
    df_metrics.loc[dataset,'Accuracy']=accuracy
    df_metrics.loc[dataset,'AUPRC']=average_precision
    df_metrics.loc[dataset,'Balanced Accuracy']=balanced_acc
    del y_probas
    gc.collect()
    return df_metrics

def plot_metrics(clsf,X,y,dataset,prgname):
    dir_plt="./plots"
    if not os.path.exists(dir_plt): os.mkdir(dir_plt)
    plot_confusion_matrix(clsf, X, y,values_format='d',display_labels=['non-interaction','interaction'])  
    plt.title('Confusion Matrix')
    plt.legend(loc='best')
    plt.savefig(dir_plt+"/"+prgname+"_conf_"+dataset+".jpg")
    plt.show() 
    plot_roc_curve(clsf, X, y)
    plt.legend(loc='best')
    plt.savefig(dir_plt+"/"+prgname+"_roc_"+dataset+".jpg")
    plt.show() 
    y_probas = clsf.predict_proba(X)[:,1]
    average_precision = average_precision_score(y, y_probas)
    disp = plot_precision_recall_curve(clsf, X, y)
    disp.ax_.set_title('Precision-Recall curve: '
                       'AP={0:0.3f}'.format(average_precision))
    plt.legend(loc='best')
    plt.savefig(dir_plt+"/"+prgname+"_auprc_"+dataset+".jpg")
    plt.show()
    del y_probas
    gc.collect()
def plot_metrics_keras(y,y_pred,y_probas,dataset,plt_dir):
    if not os.path.exists(plt_dir): os.mkdir(plt_dir)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y,y_pred), 
                              display_labels=['non-interaction','interaction'])
    disp.plot(include_values=True, cmap='viridis', 
              ax=None, xticks_rotation='horizontal',
              values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(plt_dir+"/conf_"+dataset+".jpg", pad_inches=7)
    plt.show()
    plt.close()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y, y_probas)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.plot([0, 1], [0, 1], '--', label='Random')
    plt.plot(fpr_keras, tpr_keras, label='ROC curve (AUC = {:.3f})'.format(auc_keras))
    plt.title('ROC curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right')
    plt.savefig(plt_dir+'/auroc_'+dataset+'.jpg')
    plt.show()
    plt.close()
    # calculate model precision-recall curve
    precision, recall, _ = precision_recall_curve(y, y_probas)
    # plot the model precision-recall curve
    average_precision = average_precision_score(y, y_probas)
    random_clsff = len(y[y==1]) / len(y)
    plt.plot([0, 1], [random_clsff, random_clsff], '--', label='Random')
    plt.plot(recall, precision, label='PR curve (AUC = {:.3f})'.format(average_precision))
    plt.title('Precision-Recall curve')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend(loc='best')
    plt.savefig(plt_dir+'/auprc_'+dataset+'.jpg')
    # show the plot
    plt.show()
    plt.close()
    
def df_metrics(clsf,X,y,y_pred,dataset,y_probas=''):
    metrics=[]
    metrics_X_macro=metric_results_macro(clsf,X,y,y_pred,dataset+"_macro",y_probas)
    metrics.append(metrics_X_macro)
    metrics_X_weighted=metric_results_weighted(clsf,X,y,y_pred,dataset+"_weighted",y_probas)
    metrics.append(metrics_X_weighted)
    metrics_X_0=metric_results_0(clsf,X,y,y_pred,dataset+"_0",y_probas)
    metrics.append(metrics_X_0)
    metrics_X_1=metric_results_1(clsf,X,y,y_pred,dataset+"_1",y_probas)
    metrics.append(metrics_X_1)
    df_metrics=pd.concat(metrics)
    return df_metrics
