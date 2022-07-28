# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import sys
from random import randrange
import matplotlib.pyplot as plt
import time
pd.set_option('display.max_columns', 999)

adress = 'rawdata/'
sell_data = pd.read_csv(adress + 'sale_data.csv', dtype='str')

sell_data = sell_data[[ '土地位置建物門牌', '交易標的', '建物型態', '移轉層次', '總樓層數', '主建物面積', '附屬建物面積', '陽台面積', '編號']]

######## 使用工具 ############

# 1.中文數字轉阿拉伯數字
class HanziToNumber():
    def __init__(self):
        self.CN_NUM = {
            u'〇': 0,
            u'一': 1,
            u'二': 2,
            u'三': 3,
            u'四': 4,
            u'五': 5,
            u'六': 6,
            u'七': 7,
            u'八': 8,
            u'九': 9,
        }
        self.CN_UNIT = {
            u'十': 10,
            u'拾': 10,
            u'百': 100,
        }

    def cn2dig(self, cn):
        lcn = list(cn)
        unit = 0  # 當前的單位
        ldig = []  # 臨時數組

        while lcn:
            cndig = lcn.pop()
            if cndig in self.CN_UNIT:          # python2: CN_UNIT.has_key(cndig)
                unit = self.CN_UNIT.get(cndig)
                if unit == 10000:
                    ldig.append('w')  # 標示萬位
                    unit = 1
                continue
            elif (cndig in self.CN_NUM): 
                dig = self.CN_NUM.get(cndig)
                if unit:
                    dig = dig * unit
                    unit = 0
                ldig.append(dig)

        if unit == 10:  # 處理10-19的數字
            ldig.append(10)

        ret = 0
        tmp = 0
        while ldig:
            x = ldig.pop()
            if x == 'w':
                tmp *= 10000
                ret += tmp
                tmp = 0
            else:
                tmp += x
        ret += tmp
        return ret

nth = HanziToNumber()

# 2.全形轉半型
def full_to_half(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全形空格直接轉換
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全形字元（除空格）根據關係轉化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

# 3.資料觀察
check_df = sell_data['交易標的']
value_counts = sell_data['建物型態'].value_counts(dropna=False)



######## 資料整理 ############
data = pd.DataFrame() #清好的資料，以英文統一在這

# 3.土地位置建物門牌
## 地址全形轉半型
sell_data['土地位置建物門牌'] = sell_data.apply(lambda r:full_to_half(r['土地位置建物門牌']), axis = 1)
data['address']  = sell_data['土地位置建物門牌'].copy()


# 28.主建物面積
## 面積轉換成坪 每坪=3.3058平方公尺
m2_to_yiping = 3.3058 
sell_data['主建物面積'] = sell_data['主建物面積'].astype(float) / m2_to_yiping
sell_data['主建物面積'] = sell_data['主建物面積'].round().astype(int)
data['main_area'] = sell_data['主建物面積'].copy()

# 29.附屬建物面積
## 面積轉換成坪 每坪=3.3058平方公尺
sell_data['附屬建物面積'] = sell_data['附屬建物面積'].astype(float) / m2_to_yiping
sell_data['附屬建物面積'] = sell_data['附屬建物面積'].round().astype(int)
data['ancillary_building_area'] = sell_data['附屬建物面積'].copy()

# 30.陽台面積
## 面積轉換成坪 每坪=3.3058平方公尺
sell_data[ '陽台面積'] = sell_data[ '陽台面積'].astype(float) / m2_to_yiping
sell_data[ '陽台面積'] = sell_data[ '陽台面積'].round().astype(int)
data['balcony_area'] = sell_data['陽台面積'].copy()


# 10.移轉層次、11.總轉移層

## 各種新欄位
trading_floors_count_col = [] # 交易樓層個數
building_total_floors_col = [] #建築總樓層(房子多高)
trading_floors_list_col = [] # 交易樓層清單 ex:[2,3], [1,3,4,5]
min_floors_height_col = [] # 最低交易樓層高度 ex: [2,3] => 2
including_basement_col = [] # 是否有地下室
including_arcade_col = [] # 是否有騎樓

for index in sell_data.index:
    # 建築總樓層(房子多高)
    total_floors_str = sell_data['總樓層數'][index]
    total_floors_str = str(total_floors_str)
    if(total_floors_str.isdigit()): # 判斷是否為阿拉伯數字
        total_floors = int(total_floors_str)
    else:
        total_floors = nth.cn2dig(total_floors_str) # 把文字轉成阿拉伯數字
        
    # 列出交易樓層清單 ex:[2,3], [1,3,4,5]
    trading_floors_str = str(sell_data['移轉層次'][index])
    trading_floors_num_list = [] # 以阿拉伯數字紀錄轉移樓層
    trading_floors_list = trading_floors_str.split('，')
    
    trading_floors_count = 0 # 總轉移層數
    for text in trading_floors_list:
        floor = nth.cn2dig(text) # 把文字轉成阿拉伯數字
        # print(floor)
        if(floor != 0): # 判斷是否為樓層
            trading_floors_count += 1 # 共幾層
            trading_floors_num_list.append(floor)
    
    # 從移轉層次判斷地下室、騎樓
    including_basement = 0 # 是否有地下室
    including_arcade = 0 # 是否有騎樓
    
    # 地下層沒有被算到trading_floors_count，補算
    if(trading_floors_str.find('地下層') > -1):
         trading_floors_count += 1 # 共幾層
         trading_floors_num_list.append(-1)
    
    # 紀錄是否有地下樓層
    if(trading_floors_str.find('地下')> -1):
        including_basement = 1
    
    # 紀錄是否有騎樓
    if(trading_floors_str.find('騎樓')> -1):
        including_arcade = 1
        
        
    # 最低交易樓層高度
    ## 轉移樓層清單是空的
    if(len(trading_floors_num_list) == 0):
        # 只有地下室
        if(including_basement):
            trading_floors_height = -1 # 地下室
        else:
            trading_floors_height = -2 # 之後會刪除
        # 交易層數為全時，看1樓
        if(trading_floors_str == "全"):
            trading_floors_height = 1 
    else:
        # 取最小的樓層，做轉移高度
        trading_floors_height = min(trading_floors_num_list)
    
    # 是否為透天厝
    if(sell_data['建物型態'][index] == '透天厝'):
        trading_floors_height = 1
    
    trading_floors_count_col.append( trading_floors_count)
    building_total_floors_col.append(total_floors)  
    min_floors_height_col.append(trading_floors_height)
    including_basement_col.append(including_basement)
    including_arcade_col.append(including_arcade)
    trading_floors_list_col.append(trading_floors_num_list)

    
# 新增欄位
sell_data['trading_floors_count'] = trading_floors_count_col  # 交易樓層數
sell_data['building_total_floors'] = building_total_floors_col  # 建築總樓層數(多高)
sell_data['trading_floors_list'] = trading_floors_list_col # 交易樓層清單(阿拉伯數字)
sell_data['min_floors_height'] = min_floors_height_col # 最低交易樓層高度 ex: [2,3] => 2
sell_data['including_basement'] = including_basement_col  # 是否有地下室
sell_data['including_arcade'] = including_arcade_col # 是否有騎樓




### 樓層相關(10.移轉層次、11.總轉移層)

data['trading_floors_count'] = sell_data['trading_floors_count'].copy()   # 交易樓層總數
data['building_total_floors'] = sell_data['building_total_floors'].copy()  # 建築總樓層數(多高)
data['min_floors_height'] = sell_data['min_floors_height'].copy()   # 最低交易樓層高度 ex: [2,3] => 2
data['including_basement'] = sell_data['including_basement'].copy() # 是否有地下室
data['including_arcade'] = sell_data['including_arcade'].copy() # 是否有騎樓


### 編號

data['編號'] = sell_data['編號']

# 輸出尚未刪除資料的data
data.to_csv('Output_feature/sale_data_feature_cobra.csv')


######## 排除條件 ############

data_delete = data

# 3.Adress
# 清除亂碼地址
address_garbled = ['', '', '', '', '', '', '', '', '', '', '', '', '魚', '底', '', '', '',\
         '', '', '', '', '', '', '']
data_delete = data_delete[~data_delete['address'].str[:1].isin(address_garbled)]
# 清除字數小於5的地址
data_delete = data_delete[~data_delete['address'].apply(lambda x : len(x)<5)]


# 28.主建物面積
data_delete = data_delete[data_delete['main_area'] != 0] # 刪除面積為0的資料

# 樓層
data_delete = data_delete[data_delete['min_floors_height']  != -2] # 刪除轉移樓層為空的

# 輸出刪除資料後的data
data_delete.to_csv('Output_feature/sale_data_feature_deleted_cobra.csv')
