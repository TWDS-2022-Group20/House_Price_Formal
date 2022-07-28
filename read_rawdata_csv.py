# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 22:32:00 2022

@author: User
"""

import pandas as pd
import os

adress = 'C:/Users/User/OneDrive - 銘傳大學 - Ming Chuan University/實價登陸/House_Project/rowdata/'
#adress = 'C:/Users/Z00045502/Desktop/price/'

folder = os.listdir(adress)
folder = [i for i in folder if len(i.split('.'))==1]


#file = ['A_lvr_land_A.csv', 'A_lvr_land_B.csv', 'B_lvr_land_A.csv', 'B_lvr_land_B.csv', 'C_lvr_land_A.csv', 'C_lvr_land_B.csv', 'D_lvr_land_A.csv', 'D_lvr_land_B.csv',
# 'E_lvr_land_A.csv', 'E_lvr_land_B.csv', 'F_lvr_land_A.csv', 'F_lvr_land_B.csv', 'G_lvr_land_A.csv', 'G_lvr_land_B.csv', 'H_lvr_land_A.csv', 'H_lvr_land_B.csv', 'I_lvr_land_A.csv',
# 'I_lvr_land_B.csv', 'J_lvr_land_A.csv', 'J_lvr_land_B.csv', 'K_lvr_land_A.csv', 'K_lvr_land_B.csv', 'M_lvr_land_A.csv', 'M_lvr_land_B.csv', 'N_lvr_land_A.csv', 'N_lvr_land_B.csv',
# 'O_lvr_land_A.csv', 'O_lvr_land_B.csv', 'P_lvr_land_A.csv', 'P_lvr_land_B.csv', 'Q_lvr_land_A.csv', 'Q_lvr_land_B.csv', 'T_lvr_land_A.csv', 'T_lvr_land_B.csv', 'U_lvr_land_A.csv',
# 'U_lvr_land_B.csv', 'V_lvr_land_A.csv', 'V_lvr_land_B.csv', 'W_lvr_land_A.csv', 'W_lvr_land_B.csv', 'X_lvr_land_A.csv', 'X_lvr_land_B.csv', 'Z_lvr_land_A.csv', 'Z_lvr_land_B.csv']

#adress_folder_file = []
#for i in folder:
#    for j in file:
#        adress_folder_file.append(adress + i +'/' + j)


from glob import glob
import pandas as pd

adress_folder_file = []
adress_folder_future_file = []
for i in folder:
    adress_folder_file = adress_folder_file + glob(adress + i + '/[A-Z]_*_A.csv')
    adress_folder_future_file = adress_folder_future_file + glob(adress + i + '/[A-Z]_*_B.csv')

filesize= pd.concat((pd.Series(os.path.getsize(mapper)) for mapper in adress_folder_file)).reset_index(drop = True)
filesize_future = pd.concat((pd.Series(os.path.getsize(mapper)) for mapper in adress_folder_future_file)).reset_index(drop = True)

adress_list_data = pd.concat([pd.Series(adress_folder_file),filesize],axis =1)
adress_list_data_future = pd.concat([pd.Series(adress_folder_future_file),filesize_future],axis =1)
adress_list_data.columns = ['adress','size']
adress_list_data_future.columns = ['adress','size']

using_adress = adress_list_data.query("size>0").adress
using_adress_future = adress_list_data_future.query("size>0").adress

    
mapping_df = pd.concat((pd.read_csv(mapper,dtype = 'str', on_bad_lines='warn') for mapper in using_adress),ignore_index=True)
mapping_future_df = pd.concat((pd.read_csv(mapper,dtype = 'str', on_bad_lines='warn') for mapper in using_adress_future),ignore_index=True)

mapping_df = mapping_df.query("鄉鎮市區!='The villages and towns urban district'")
mapping_future_df = mapping_future_df.query("鄉鎮市區!='The villages and towns urban district'")

mapping_df.to_csv(adress + 'sale_data.csv',index = False)
mapping_future_df.to_csv(adress + 'sale_future_data.csv',index = False)
