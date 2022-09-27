# -*- coding: utf-8 -*-
"""
@author: engin aybey
"""
import pandas as pd
from sklearn.utils import resample

def down_dataset_1_1(df):

    df_no_interaction =df[df['Interaction']==0]
    df_interaction =df[df['Interaction']==1]
    
    df_no_interaction_downsampled = resample(df_no_interaction,
                                        replace=False,
                                        n_samples=len(df_interaction),
                                        random_state=42)
    len(df_no_interaction_downsampled)
    
    df_interaction_downsampled = resample(df_interaction,
                                        replace=False,
                                        n_samples=len(df_interaction),
                                        random_state=42)
    len(df_interaction_downsampled)
    
    df_downsample=pd.concat([df_no_interaction_downsampled,df_interaction_downsampled])
    len(df_downsample)
    return df_downsample