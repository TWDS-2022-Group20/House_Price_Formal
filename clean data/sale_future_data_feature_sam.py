#!/usr/bin/env python
# coding: utf-8
# By Sam

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)
import matplotlib.pyplot as plt
import os
import glob
import re

# set the dir path before execute
dir_path = 'D:\programming\TWDS\House_price'
file_path = os.path.join(dir_path, 'sale_future_data.csv')
raw_df = pd.read_csv(file_path)

# 將指定欄位篩選出來
clean_df_future = raw_df.copy(deep=True)
clean_df_future = clean_df_future[ ['編號', '都市土地使用分區', '非都市土地使用編定', '交易筆棟數', '備註'] ]

################################
######### Delete data ##########
################################
clean_df_future = clean_df_future[ clean_df_future['交易筆棟數'] != '土地0建物0車位0' ]


################################
######## Data cleaning #########
################################

##### 非都市土地使用編定 #####

# 在'都市土地使用分區'欄位中，誤植'非都市資料'，將誤植的資料移入正確位置('非都市土地使用編定')
mis_df = raw_df[ raw_df['都市土地使用分區'].apply(lambda x: str(x).startswith('非都市')) & raw_df['非都市土地使用分區'].isna()]
mis_df = mis_df[~ mis_df['都市土地使用分區'].apply(lambda x: str(x).startswith('非都市： ;')) ]
clean_df_future.loc[mis_df.index, '非都市土地使用編定'] = clean_df_future.loc[mis_df.index,'都市土地使用分區'].apply(lambda x: x.split('：')[-1])

# 將'非都市土地使用編定'中非空值的部分轉換成獨立的類別欄位，空值則忽略不計(若其餘類別欄位為0則代表空值)
# for one-hot encoding
non_city_code = {'農牧用地':'Non_City_Farm', 
                 '甲種建築用地':'Non_City_FirstBuild', 
                 '乙種建築用地':'Non_City_SecondBuild', 
                 '丙種建築用地':'Non_City_ThirdBuild', 
                 '丁種建築用地':'Non_City_FourthBuild',
                 '暫未編定用地':'Non_City_Pending', 
                 '暫未編定':'Non_City_Pending', 
                 '林業用地':'Non_City_Forest', 
                 '殯葬用地':'Non_City_Funeral', 
                 '交通用地':'Non_City_Traffic', 
                 '特定目的事業用地':'Non_City_Special', 
                 '水利用地':'Non_City_Hydraulic', 
                 '遊憩用地':'Non_City_Recreation', 
                 '養殖用地':'Non_City_Culture', 
                 '窯業用地':'Non_City_Ceramic',
                 '生態保護用地':'Non_City_Eco', 
                 '國土保安用地':'Non_City_NationSecure', 
                 '古蹟保存用地':'Non_City_Historical', 
                 '礦業用地':'Non_City_Mining', 
                 '墳墓用地':'Non_City_Tomb', 
                 '鹽業用地':'Non_City_Salt'}
# for key, value in non_city_code.items():
#     clean_df[value] = (clean_df['非都市土地使用編定'] == key).astype('int')

# for label encoding
non_city_label = {'農牧用地':1, 
                 '甲種建築用地':2, 
                 '乙種建築用地':3, 
                 '丙種建築用地':4, 
                 '丁種建築用地':5,
                 '暫未編定用地':6, 
                 '暫未編定':7, 
                 '林業用地':8, 
                 '殯葬用地':9, 
                 '交通用地':10, 
                 '特定目的事業用地':11, 
                 '水利用地':12, 
                 '遊憩用地':13, 
                 '養殖用地':14, 
                 '窯業用地':15,
                 '生態保護用地':16, 
                 '國土保安用地':17, 
                 '古蹟保存用地':18, 
                 '礦業用地':19, 
                 '墳墓用地':20, 
                 '鹽業用地':21}
clean_df_future['非都市土地使用編定'].fillna(0, inplace=True)
clean_df_future['非都市土地使用編定'] = clean_df_future['非都市土地使用編定'].replace(non_city_label)

##### 交易筆棟數 #####

# '交易筆棟數': '土地?建物?車位?'，將其字串格式轉換為 3 columns with int dtype
# '交易筆棟數'欄位並無空值，但有出現'土地0建物0車位0'的值，共50筆

clean_df_future['Transaction_Land'] = clean_df_future['交易筆棟數'].apply(lambda x: int(re.findall(r'.{2}[0-9]{1}', x)[0][-1]))
clean_df_future['Transaction_Building'] = clean_df_future['交易筆棟數'].apply(lambda x: int(re.findall(r'.{2}[0-9]{1}', x)[1][-1]))
clean_df_future['Transaction_Parking'] = clean_df_future['交易筆棟數'].apply(lambda x: int(re.findall(r'.{2}[0-9]{1}', x)[2][-1]))

##### 備註 #####
# 將'備註'欄位ˋ中的空值進行轉換，再查找關鍵字後建立對應的類別欄位，空值有考慮(其餘類別欄位為0不一定代表無字串)
clean_df_future['備註'].fillna('無備註', inplace=True)
note_dic = {
    'Note_Null':['無備註'],
    'Note_Additions':['含增建', '未登記建物', '其他增建'], # 增建或未登記建物
    'Note_Presold':['預售屋'], # 預售屋
    'Note_Relationships':['特殊關係', '親友', '親屬', '親等', '等親', '近親', '關係人', '朋友', '姐弟', '姊弟', '夫妻', '兄弟', '母子', '父子', '同學', '共有人間', '承租人間'],
    'Note_Balcony':['陽台'], # 陽台外推
    'Note_PublicUtilities':['公共設施保留地'], # 公共設施保留地
    'Note_PartRegister':['分次登記案件'], # 分次登記案件
    'Note_Negotiate':['協議價購'], # 協議價購
    'Note_Parking':['車位'], # 含車位
    'Note_OnlyParking':['僅車位', '單獨車位交易'], # 僅車位交易
    'Note_Gov':['政府機關'], # 政府機關承購
    'Note_Overbuild':['頂樓'], # 頂樓加蓋
    'Note_Decoration':['含裝潢', '附裝潢', '含裝修', '裝潢'], # 裝潢
    'Note_Furniture':['含傢俱', '含家電','家電'], # 傢俱
    'Note_Layer':['夾層'], # 夾層
    'Note_BuildWithLandholder':['建商與地主合建'], # 建商與地主合建案
    'Note_BlankHouse':['毛胚屋'], # 毛胚屋
    'Note_Defect':['瑕疵', '有難', '凶宅', '兇宅'], # 瑕疵
    'Note_Debt':['債權債務', '債務抵償'], # 債權債務
    'Note_Elevator':['附電梯', '有電梯', '含電梯'], # 電梯
    'Note_Renewal':['都更'], # 都更效益
    'Note_DistressSale ':['急賣'], # 急賣
    'Note_OverdueInherit':['未辦繼承'], # 逾期未辦繼承
    'Note_DeformedLand':['畸零地'], # 畸零地
    'Note_Shop':['店鋪', '店面'] # 店鋪
}

def note_features(note_key, text):
    value = 0
    for word in note_dic[note_key]:
        if word in text:
            value = 1
    return value

for key in note_dic:
    clean_df_future[key] = clean_df_future['備註'].apply(lambda x: note_features(key, x))

################################
#### Rename and output csv #####
################################


# 最後將多餘欄位drop，並將欄位名轉換成英文
clean_df_future.drop(columns=['都市土地使用分區', '備註', '交易筆棟數'], inplace=True)
columns_map_sam = {
    '非都市土地使用編定':'Non_City_Land_Code',
}
clean_df_future.rename(columns=columns_map_sam, inplace=True)
# export the dataframe into csv file
output_path = os.path.join(dir_path, 'sale_future_data_feature_sam.csv')
clean_df_future.to_csv(output_path)
