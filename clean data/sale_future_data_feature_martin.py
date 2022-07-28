# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:54:35 2022

@author: Z00045502
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

sell_data = pd.read_csv(adress + 'rawdata/sale_future_data.csv', dtype='str')
sell_data.loc[:,'鄉鎮市區'] = sell_data['鄉鎮市區'].apply(lambda x : str(x).replace('台','臺')) # 文字統一
sell_data.loc[:,'鄉鎮市區'] = sell_data['鄉鎮市區'].apply(lambda x : str(x).replace('巿','市')) # 文字統一
sell_data.loc[:,'鄉鎮市區'] = sell_data['鄉鎮市區'].apply(lambda x : str(x).replace('金fa4b鄉','金峰鄉'))
sell_data.loc[:,'鄉鎮市區'] = sell_data['鄉鎮市區'].apply(lambda x : str(x).replace('fa72埔鄉','鹽埔鄉'))
sell_data.loc[sell_data['鄉鎮市區']=='nan','鄉鎮市區'] = '竹東鎮'


place_id = pd.read_csv(adress + 'place_id.csv', dtype='str',encoding='big5')
place_id['place_detail'] = place_id.place.str[3:]
place_id.loc[place_id.place_detail=='','place_detail'] = place_id.loc[place_id.place_detail=='','place']
mul_place_detail = place_id.place_detail.value_counts()[place_id.place_detail.value_counts().sort_values()>1].index
mul_place_id = place_id[place_id.place_detail.isin(mul_place_detail)]
mul_place_id['place_detail'] = mul_place_id.place.str[:3] + mul_place_id.place_detail
place_id = place_id[~place_id.place_detail.isin(mul_place_detail)]
place_id  = pd.concat([mul_place_id,place_id])

# a = sell_data.merge(place_id , how = 'left' , left_on = '鄉鎮市區' , right_on = 'place_detail')

# 排除條件
# 1.土地交易
sell_data = sell_data[(sell_data['交易標的']!='土地') & (~sell_data['交易標的'].isna())]
# 2. 交易年月日調整，剔除成交月份不再106

sell_data['Month'] = sell_data['交易年月日'].str[:-2].astype('float')
sell_data = sell_data.query("Month>=10601 and Month<=11103 ")
sell_data = sell_data[(sell_data.Month!=10600) & (sell_data.Month!=10700) & (sell_data.Month!=10800) & (sell_data.Month!=10900) & (sell_data.Month!=11000)]
# a = sell_data[(sell_data.Month<10601) | (sell_data.Month>11003)]

# 3.建築完成年月 全部皆為NA，因此剃除
#sell_data = sell_data[~sell_data['建築完成年月'].isna()]
#sell_data = sell_data[sell_data['建築完成年月'].str.isdigit()]
#sell_data = sell_data[sell_data['建築完成年月'].apply(lambda x : len(x)==7)] # 判斷是否有7碼
#sell_data = sell_data[sell_data['建築完成年月'].str[:3].astype('float')!=0] # 判斷年份有資料
#sell_data = sell_data[(sell_data['建築完成年月'].str[3:5].astype('float')>=(1)) & (sell_data['建築完成年月'].str[3:5].astype('float')<=12)] # 判斷月份有資料
#sell_data = sell_data[(sell_data['建築完成年月'].str[5:7].astype('float')>=(1)) & (sell_data['建築完成年月'].str[5:7].astype('float')<=31)] # 判斷日有資料
#sell_data = sell_data[(sell_data['建築完成年月'].str[3:7].astype('float')!=229) & (sell_data['建築完成年月'].str[3:7].astype('float')!=230) & (sell_data['建築完成年月'].str[3:7].astype('float')!=431) & 
#                      (sell_data['建築完成年月'].str[3:7].astype('float')!=631) & (sell_data['建築完成年月'].str[3:7].astype('float')!=931) & (sell_data['建築完成年月'].str[3:7].astype('float')!=1131)]
#sell_data = sell_data.drop([151033,1515858]).sort_values("建築完成年月") # 刪除年份看起來很奇怪的部分

# '建物現況格局-房' 經人員檢測10間以上為極端值
sell_data = sell_data[sell_data['建物現況格局-房'].astype("float")<=10]

# '建物現況格局-廳' 經人員檢測22間以上為極端值
sell_data = sell_data[sell_data['建物現況格局-廳'].astype("float")<22]

# '建物現況格局-衛' 經人員檢測12間以上為極端值
sell_data = sell_data[sell_data['建物現況格局-衛'].astype("float")<12]

# 總價元 需不為0
sell_data = sell_data[sell_data['總價元'].astype("float")!=0]

# '車位總價元'→用車位類別做為拆選條件
# sell_data = sell_data[sell_data['車位總價元'].astype("float")!=0]

# 重複的地區須結合地址
mul_sell_data = sell_data[sell_data['鄉鎮市區'].isin(mul_place_detail)]
mul_sell_data['鄉鎮市區'] = mul_sell_data['土地位置建物門牌'].str[:3] + mul_sell_data['鄉鎮市區']
sell_data = sell_data[~sell_data['鄉鎮市區'].isin(mul_place_detail)]
sell_data = pd.concat([sell_data,mul_sell_data ])


# 鄉鎮市區
data = pd.DataFrame()
data['Place_id'] = sell_data['鄉鎮市區'].copy()
data['Place_id'] = data['Place_id'].map(dict(zip(place_id.place_detail, place_id.place_id)))
print(data[data.Place_id.isna()])

# 交易標的
data['Type'] = sell_data['交易標的'].copy()
# {'房地(土地+建物)': 0, '建物': 1, '房地(土地+建物)+車位': 2, '車位': 3}
# print(dict(zip(sell_data['交易標的'].unique(), range(len(data.Type.unique())))))
data['Type'] = data['Type'].map({'房地(土地+建物)': 0, '建物': 1, '房地(土地+建物)+車位': 2, '車位': 3})
print(data[data.Type.isna()])



# 門牌編號

#GM_F = GM()
## GM_F.coordination(url='https://www.google.com.tw/maps/place?q=臺北市中正區門街９９巷３５之３號二樓')

#URL = sell_data['土地位置建物門牌'].copy()

#d = 0
#ws = []
#url_append = []
#for i in URL:
#    url_list= 'https://www.google.com.tw/maps/place?q=' + i
#    if d>100:
#        time.sleep(60)
#        d = 0
#    else:
#        time.sleep(randrange(5))
#        d = d+1
#    ws.append(GM_F.coordination(url = url_list))
#    url_append.append(i)

#adress = pd.concat([pd.DataFrame(ws),pd.DataFrame(url_append)],axis= 1)
#adress.columns = ['latitude','longitude','adress']

#data['adress'] = sell_data['土地位置建物門牌'].copy()
#data = data.merge(adress, how = 'left', on = 'adress')
#data = data.drop('adress',axis = 1)


## 土地移轉總面積平方公尺
data['area_m2'] = sell_data['土地移轉總面積平方公尺'].copy().astype('float')
data['area_ping'] = round(data.area_m2 / 3.3058,0)
data[data.area_m2.isna()]

## 都市土地使用分區、非都市土地使用分區、非都市土地使用編定 先不使用
## 交易日
data['TDATE'] = sell_data['交易年月日'].copy()
data['TDATE'] = pd.to_datetime(data['TDATE'].astype('int') + 19110000,format = '%Y%m%d')
data['Month'] = data.TDATE.dt.strftime('%Y%m')

## 建築完成年月 全部皆為NA，因此剃除
#data['Finish_Date'] = sell_data['建築完成年月'].copy()
#data['Finish_Date'] = pd.to_datetime(data['Finish_Date'].astype('int') + 19110000,format = '%Y%m%d')
#data['Finish_Month'] = data.Finish_Date.dt.strftime('%Y%m')


## 屋齡
#data['house_age'] = round((data['TDATE'] - data['Finish_Date']).dt.days/365,0)

## '建物現況格局-房'
data['room'] = sell_data['建物現況格局-房'].copy().astype('float')

## 建物現況格局-廳'
data['hall'] = sell_data['建物現況格局-廳'].copy().astype('float')

## '建物現況格局-衛'
data['bathroom'] = sell_data['建物現況格局-衛'].copy().astype('float')

## '建物現況格局-隔間'
data['compartment'] = sell_data['建物現況格局-隔間'].copy()
data['compartment'] = data['compartment'].map(dict(zip(['有','無'], [1,0])))

## '有無管理組織'
data['manager'] = sell_data['有無管理組織'].copy()
data['manager'] = data['manager'].map(dict(zip(['有','無'], [1,0])))

## 總價元
data['Total_price'] = sell_data['總價元'].copy().astype("float")

# 車位總價元
data['parking_price'] = sell_data['車位總價元'].copy().astype("float")

# 編號【key】
data['編號'] = sell_data['編號'].copy()


# output data
data.to_csv(adress + 'output_feature/sale_future_data_feature_martin.csv')
