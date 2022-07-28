# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:57:27 2022

@author: User
"""


import pandas as pd
import numpy as np
import sys
sys.path.append('C:/Users/User/OneDrive - 銘傳大學 - Ming Chuan University/實價登陸/House_Project')
from google_map import google_api_module as GM
from random import randrange
import time
pd.set_option('display.max_columns', 999)

adress = 'C:/Users/User/OneDrive - 銘傳大學 - Ming Chuan University/實價登陸/House_Project/'

## get all data 

martin = pd.read_csv(adress + 'output_feature/sale_data_feature_martin.csv', dtype='str')
martin = martin.drop(['Unnamed: 0'],axis = 1)
corbra  = pd.read_csv(adress + 'output_feature/sale_data_feature_cobra.csv', dtype='str')
corbra = corbra.drop(['Unnamed: 0'],axis = 1)
eli = pd.read_csv(adress + 'output_feature/sale_data_feature_elichen.csv', dtype='str')
eli = eli.drop(['Unnamed: 0'],axis = 1)
sam = pd.read_csv(adress + 'output_feature/sale_data_feature_sam.csv', dtype='str')   
sam = sam.drop(['Unnamed: 0'],axis = 1)
allen = pd.read_csv(adress + 'output_feature/sale_data_feature_allen.csv', dtype='str')    
allen = allen.drop(['Unnamed: 0'],axis = 1)
allen.columns = ['Non_City_Land_Usage', 'Parking_Space_Types', 'Building_Types','Unit_Price_Ping', 'Transfer_Total_Ping', '編號']

data = martin.merge(corbra, how = 'inner').merge(eli, how = 'inner').merge(sam, how = 'inner').merge(allen, how = 'inner')


## combine ecnomic data
ecnomic = pd.read_csv(adress + 'output_feature/economic_rate.csv', dtype='str')    
data = data.merge(ecnomic,how = 'left', on = 'Month')


for i in data.columns:
    if data[str(i)].isna().any():
        print(i)
        
        
        
data.query("Month!='202201' and Month!='202202' and Month!='202203'").to_csv(adress + 'output_feature/clean_data_train.csv',index = False)        
data.query("Month=='202201' or Month=='202202' or Month=='202203'").to_csv(adress + 'output_feature/clean_data_test.csv',index = False)

data.Month.sort_values().unique()
