# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as np


#%%

# import data
sell_data = pd.read_csv(r"D:\download\sale_data.csv")
sell_future_data = pd.read_csv(r"D:\download\sale_future_data.csv")

#sell_data[['非都市土地使用分區',  '車位類別', '建物型態', '單價元平方公尺', '建物移轉總面積平方公尺']]


#%%
'''
# 非都市土地使用分區
1. 將與 非都市土地使用編定、都市土地使用分區 重疊紀錄的部分 encoding 0 (給都市 or 非編定)
2. encoding 分區種類
3. 刪除沒有編訂入 都市使用分區 非都市使用分區 非都市使用編定 的資料
'''
# sell_data
sell_data[['非都市土地使用編定','非都市土地使用分區','都市土地使用分區']] = sell_data[['非都市土地使用編定','非都市土地使用分區','都市土地使用分區']].fillna(0)

sell_data['非都市土地使用分區'] = sell_data.apply(lambda df: 0 if (df['非都市土地使用編定']!=0) and 
                                                                (df['非都市土地使用分區']!=0) and 
                                                                (df['都市土地使用分區']!=0)
                                                                else df['非都市土地使用分區'], axis=1)

sell_data['非都市土地使用分區'] = sell_data.apply(lambda df: 0 if (df['非都市土地使用編定']==0) and 
                                                                (df['非都市土地使用分區']!=0) and 
                                                                (df['都市土地使用分區']!=0)
                                                                else df['非都市土地使用分區'], axis=1)

sell_data['非都市土地使用分區'] = sell_data.apply(lambda df: 0 if (df['非都市土地使用編定']!=0) and 
                                                                (df['非都市土地使用分區']!=0) and 
                                                                (df['都市土地使用分區']==0)
                                                                else df['非都市土地使用分區'], axis=1)
# encoding 
a = sell_data['非都市土地使用分區'].value_counts().index
sell_data['非都市土地使用分區'] = sell_data['非都市土地使用分區'].replace(a, np.arange(0, len(a)))
                                                                       

# sell_future_data
sell_future_data[['非都市土地使用編定','非都市土地使用分區','都市土地使用分區']] = sell_future_data[['非都市土地使用編定','非都市土地使用分區','都市土地使用分區']].fillna(0)

sell_future_data['非都市土地使用分區'] = sell_future_data.apply(lambda df: 0 if (df['非都市土地使用編定']!=0) and 
                                                                (df['非都市土地使用分區']!=0) and 
                                                                (df['都市土地使用分區']!=0)
                                                                else df['非都市土地使用分區'], axis=1)

sell_future_data['非都市土地使用分區'] = sell_future_data.apply(lambda df: 0 if (df['非都市土地使用編定']==0) and 
                                                                (df['非都市土地使用分區']!=0) and 
                                                                (df['都市土地使用分區']!=0)
                                                                else df['非都市土地使用分區'], axis=1)

sell_future_data['非都市土地使用分區'] = sell_future_data.apply(lambda df: 0 if (df['非都市土地使用編定']!=0) and 
                                                                (df['非都市土地使用分區']!=0) and 
                                                                (df['都市土地使用分區']==0)
                                                                else df['非都市土地使用分區'], axis=1)
# encoding 
b = sell_future_data['非都市土地使用分區'].value_counts().index
sell_future_data['非都市土地使用分區'] = sell_future_data['非都市土地使用分區'].replace(b, np.arange(0, len(b)))

#%%
# 刪除無編訂的資料 (24,319)
sell_data.drop(sell_data[(sell_data['非都市土地使用編定']==0) & (sell_data['非都市土地使用分區']==0) & (sell_data['都市土地使用分區']==0)].index, inplace=True)
sell_future_data.drop(sell_future_data[(sell_future_data['非都市土地使用編定']==0) & (sell_future_data['非都市土地使用分區']==0) & (sell_future_data['都市土地使用分區']==0)].index, inplace=True)

#%%
'''
# 車位類別
1. NaN 代表無車位
2. 將 sell_data 車位類別 encoding
3. 將 sell_future_data 車位類別 encoding
'''
# Fill NaN with 無車位 (sell_data)
sell_data['車位類別'] = sell_data['車位類別'].fillna("無車位")
# Fill NaN with 無車位 (sell_data)
sell_future_data['車位類別'] = sell_future_data['車位類別'].fillna("無車位")

# encoding 車位類別 (sell_data)
c = sell_data['車位類別'].value_counts().index
sell_data['車位類別'] = sell_data['車位類別'].replace(c, np.arange(0,len(c)))
                                                    
# encoding 車位類別 (sell_future_data)
d = sell_future_data['車位類別'].value_counts().index
sell_future_data['車位類別'] = sell_future_data['車位類別'].replace(d, np.arange(0,len(d)))

#%%
'''
# 建物型態
1. 將 sell_data 建物型態 encoding
2. 將 sell_future_data 建物型態 encoding (sell_future_data 沒有倉庫與農舍)
'''
# encoding 建物型態 (sell_data)
e = sell_data['建物型態'].value_counts().index
sell_data['建物型態'] = sell_data['建物型態'].replace(e, np.arange(0,len(e)))
                                                    
# encoding 建物型態 (sell_future_data)
f = sell_future_data['建物型態'].value_counts().index
sell_future_data['建物型態'] = sell_future_data['建物型態'].replace(f, np.arange(0,len(f)))

#%%

'''
# 建物移轉總面積平方公尺
1. Fill NaN with 0
2. 計算建物移轉總面積平方公尺 = 0 的值 (建物移轉總面積平方公尺 = 總價元 / 單位元平方公尺)
3. 刪除掉無法換算的資料
'''
# Fill NaN with 0
sell_data[['總價元','建物移轉總面積平方公尺','單價元平方公尺']] = sell_data[['總價元','建物移轉總面積平方公尺','單價元平方公尺']].fillna(0)
sell_future_data[['總價元','建物移轉總面積平方公尺','單價元平方公尺']] = sell_future_data[['總價元','建物移轉總面積平方公尺','單價元平方公尺']].fillna(0)

# 計算 建物移轉總面積平方公尺 (sell_data)
sell_data['建物移轉總面積平方公尺'] = sell_data.apply(lambda df: df['總價元']/df['單價元平方公尺'] if (df['總價元']!=0) and 
                                                                                                  (df['建物移轉總面積平方公尺']==0) and 
                                                                                                  (df['單價元平方公尺']!=0)
                                                                                                  else df['建物移轉總面積平方公尺'], axis=1)

# 計算 建物移轉總面積平方公尺 (sell_future_data)
sell_future_data['建物移轉總面積平方公尺'] = sell_future_data.apply(lambda df: df['總價元']/df['單價元平方公尺'] if (df['總價元']!=0) and 
                                                                                                                (df['建物移轉總面積平方公尺']==0) and 
                                                                                                                (df['單價元平方公尺']!=0)
                                                                                                                else df['建物移轉總面積平方公尺'], axis=1)
#%%
# 刪除無法換算的資料 (sell_data)
sell_data.drop(sell_data[sell_data['建物移轉總面積平方公尺']==0].index, inplace=True)

# 刪除無法換算的資料 (sell_future_data)
sell_future_data.drop(sell_future_data[sell_future_data['建物移轉總面積平方公尺']==0].index, inplace=True)

#%%
'''
# 單價元平方公尺 
1. 計算單位元平方公尺 = 0 或 NaN 的值 (單位元平方公尺 = 總價元 / 建物移轉總面積平方公尺)
2. 刪除無法轉換的資料

'''
# 計算單位元平方公尺 (sell_data)
sell_data['單價元平方公尺'] = sell_data.apply(lambda df: df['總價元']/df['建物移轉總面積平方公尺'] if(df['總價元']!=0) and 
                                                                                                  (df['建物移轉總面積平方公尺']!=0) and 
                                                                                                  (df['單價元平方公尺']==0)
                                                                                                  else df['單價元平方公尺'], axis=1)

# 計算單位元平方公尺 (sell_future_data)
sell_future_data['單價元平方公尺'] = sell_future_data.apply(lambda df: df['總價元']/df['建物移轉總面積平方公尺'] if(df['總價元']!=0) and 
                                                                                                  (df['建物移轉總面積平方公尺']!=0) and 
                                                                                                  (df['單價元平方公尺']==0)
                                                                                                  else df['單價元平方公尺'], axis=1)
#%%
# 刪除無法換算的資料 (sell_data)
sell_data.drop(sell_data[sell_data['單價元平方公尺']==0].index, inplace=True)

# 刪除無法換算的資料 (sell_future_data)
sell_future_data.drop(sell_future_data[sell_future_data['單價元平方公尺']==0].index, inplace=True)


#%%
'''
# output data
1. 換算 建物移轉總面積平方公尺 → 建物移轉總面積坪
2. 刪除 建物移轉總面積坪 = 0 的資料 (太小)
3. 換算 單價元平方公尺 → 單價元坪
4. 中文特徵改為英文
5. sell_data 存入 data 的 DataFrame
'''


# output dataframe
sell_data = sell_data[['非都市土地使用分區',  '車位類別', '建物型態', '單價元平方公尺', '建物移轉總面積平方公尺','編號']]
sell_future_data = sell_future_data[['非都市土地使用分區',  '車位類別', '建物型態', '單價元平方公尺', '建物移轉總面積平方公尺','編號']]


# 換算 建物移轉總面積坪
sell_data['建物移轉總面積平方公尺'] =  (sell_data['建物移轉總面積平方公尺']/3.3058).round(0)
sell_future_data['建物移轉總面積平方公尺'] =  (sell_future_data['建物移轉總面積平方公尺']/3.3058).round(0)

#%%
# 刪除 建物移轉總面積坪 = 0 的資料
sell_data.drop(sell_data[sell_data['建物移轉總面積平方公尺']==0].index, inplace=True)
sell_future_data.drop(sell_future_data[sell_future_data['建物移轉總面積平方公尺']==0].index, inplace=True)
#%%
# 換算 單價元坪
sell_data['單價元平方公尺'] =  sell_data['單價元平方公尺']*3.3058
sell_future_data['單價元平方公尺'] =  sell_future_data['單價元平方公尺']*3.3058
#%%
# 欄位名稱英文轉換
sell_data.rename(columns={'非都市土地使用分區':'Non_City_Land_Usage', 
                          '車位類別':'Parking_Space_Types',
                          '建物型態':'Building_Types',
                          '單價元平方公尺':'Unit_Price_Ping',
                          '建物移轉總面積平方公尺':'Transfer_Total_Ping',
                          '編號':'Key'}, inplace=True)

sell_future_data.rename(columns={'非都市土地使用分區':'Non_City_Land_Usage', 
                          '車位類別':'Parking_Space_Types',
                          '建物型態':'Building_Types',
                          '單價元平方公尺':'Unit_Price_Ping',
                          '建物移轉總面積平方公尺':'Transfer_Total_Ping',
                          '編號':'Key'}, inplace=True)


data = pd.DataFrame()
#%%
sell_data.to_csv(r'D:\download\sale_data_feature_allen(full).csv')
sell_future_data.to_csv(r'D:\download\sale_future_data_feature_allen(full).csv')

# %%
