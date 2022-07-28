# -*- coding: utf-8 -*-
"""
Created on Sun May 29 13:42:12 2022

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
train = pd.read_csv(adress + 'output_feature/clean_data_train.csv',dtype = 'str')
train_future = pd.read_csv(adress + 'output_feature/clean_data_future_train.csv',dtype = 'str')

train_future['house_age'] = -1
train_future['main_area'] = train_future.area_ping.copy()


use_list = train.columns[train.columns.isin(train_future.columns)].values

train_all = pd.concat([train[use_list],train_future[use_list]])

test =  pd.read_csv(adress + 'output_feature/clean_data_test.csv',dtype = 'str')
test_future = pd.read_csv(adress + 'output_feature/clean_data_future_test.csv',dtype = 'str')

test_future['house_age'] = -1
test_future['main_area'] = test_future.area_ping.copy()

test_all = pd.concat([test[use_list],test_future[use_list]])

#train_all.to_csv(adress + 'output_feature/clean_data_all_train.csv',index = False)
#test_all.to_csv(adress + 'output_feature/clean_data_all_test.csv',index = False)

train_all.to_csv(adress + 'output_feature/clean_data_all_add_variable_train.csv',index = False)
test_all.to_csv(adress + 'output_feature/clean_data_all_add_variable_test.csv',index = False)
