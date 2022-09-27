# -*- coding: utf-8 -*-
"""
@author: engin aybey
"""
import sys
from sklearn.metrics import classification_report,confusion_matrix
def writeclassreport(y_cs,y_pred,title,output_file):
    orig_stdout = sys.stdout
    f = open(output_file, 'w+')
    sys.stdout = f
    print(title)
    print(confusion_matrix(y_cs,y_pred))
    print(classification_report(y_cs,y_pred,digits=3))
    sys.stdout = orig_stdout
    f.close()

def writeclassreporta(y_cs,y_pred,title,output_file):
    orig_stdout = sys.stdout
    f = open(output_file, 'a')
    sys.stdout = f
    print(title)
    print(confusion_matrix(y_cs,y_pred))
    print(classification_report(y_cs,y_pred,digits=3))
    sys.stdout = orig_stdout
    f.close()