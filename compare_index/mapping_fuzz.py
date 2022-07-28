# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:32:51 2022

@author: User
"""

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
import sys

import difflib

adress = 'C:/Users/User/OneDrive - 銘傳大學 - Ming Chuan University/實價登陸/House_Project/'
sys.path.append(adress)
from eval import simple_evaluate,evaluate_partitions,default_partitions

TRAIN_DATA_PATH = adress + 'output_feature/clean_data_all_add_variable_train.csv'
TEST_DATA_PATH = adress + 'output_feature/clean_data_all_add_variable_test.csv'

raw_data = pd.read_csv(TRAIN_DATA_PATH ,dtype = 'str')
raw_data_test = pd.read_csv(TEST_DATA_PATH ,dtype = 'str')

use_column = ['Place_id','Type','Transfer_Total_Ping','Building_Types','house_age','min_floors_height']


str_columns = ['Place_id','Type','compartment','manager','including_basement','including_arcade',
               'City_Land_Usage','Main_Usage_Walk','Main_Usage_Living','Main_Usage_Selling',
               'Main_Usage_Manufacturing','Main_Usage_Business','Main_Usage_Parking','Main_Usage_SnE',
               'Main_Usage_Farm','Building_Material_S','Building_Material_R','Building_Material_C',
               'Building_Material_steel','Building_Material_stone','Building_Material_B','Building_Material_W',
               'Building_Material_iron','Building_Material_tile','Building_Material_clay','Building_Material_RC_reinforce',
               'Non_City_Land_Code','Note_Null','Note_Additions','Note_Presold','Note_Relationships', #'Elevator',
               'Note_Balcony','Note_PublicUtilities','Note_PartRegister','Note_Negotiate','Note_Parking',
               'Note_OnlyParking','Note_Gov','Note_Overbuild','Note_Decoration','Note_Furniture','Note_Layer',
               'Note_BuildWithLandholder','Note_BlankHouse','Note_Defect','Note_Debt','Note_Elevator',
               'Note_Renewal','Note_DistressSale ','Note_OverdueInherit','Note_DeformedLand','Non_City_Land_Usage',
               'Parking_Space_Types','Building_Types',]
float_columns = ['area_m2','area_ping','house_age','room','hall','bathroom','Total_price',
                 'parking_price','main_area','ancillary_building_area','balcony_area','trading_floors_count',
                 'min_floors_height','building_total_floors','Parking_Area','Transaction_Land','Transaction_Building',
                 'Transaction_Parking','Unit_Price_Ping','Transfer_Total_Ping','CPI','CPI_rate','unemployment rate',
                 'Pain_index_3month','ppen_price','high_price','low_price','close_price','qmatch','amt_millon',
                 'return_rate_month','Turnover_rate_month','outstanding_share_thousand','Capitalization_million',
                 'excess total _ million_usdollars','import_price_index_usdollars','export_price_index_usdollars',
                 'export_million_usdollars','import_million_usdollars','survival_mobility_rate','live_deposit_mobility_interest_rate',
                 'CCI_3month','construction_engineering_index']
time_columns = ['TDATE','Month','Finish_Date','Finish_Month','Month_raw']
ID_columns = ['編號','address']
str_columns = pd.Series(str_columns)[pd.Series(str_columns).isin(raw_data.columns)].tolist()
float_columns = pd.Series(float_columns)[pd.Series(float_columns).isin(raw_data.columns)].tolist()

def clean_data(raw_data):
        
    
    raw_data[str_columns] = raw_data[str_columns].astype('str')
    raw_data[float_columns] = raw_data[float_columns].astype('float')
    raw_data.TDATE = pd.to_datetime(raw_data.TDATE)
    #raw_data.Finish_Date = pd.to_datetime(raw_data.Finish_Date)
    raw_data[['Month','Month_raw']] = raw_data[['Month','Month_raw']].astype("int")
    raw_data.Month = raw_data.Month.astype('int')
    
    raw_data = raw_data.drop(['編號','address','TDATE','Month_raw','parking_price'],axis = 1) #,'Finish_Date','Finish_Month'
    
    return raw_data

## Eli clean data
def clean_and_drop(df):
    # 只篩選有包含 '住' 用途的交易案
    df = df.loc[df['Main_Usage_Living'] == 1]
    df = df.drop(columns=['Main_Usage_Living'])

    # 因為都是 0
    df = df.drop(columns=['Non_City_Land_Usage', 'Main_Usage_Walk',
                          'Main_Usage_Selling',
                          'Main_Usage_SnE'])

    # 只有 344 筆是包含工廠用途，且都不具住宅用途，故剔除
    df = df.loc[df['Main_Usage_Manufacturing'] == 0]
    df = df.drop(columns=['Main_Usage_Manufacturing'])

    # 只有 76 筆是包含停車用途，且都不具住宅用途，故剔除
    df = df.loc[df['Main_Usage_Parking'] == 0]
    df = df.drop(columns=['Main_Usage_Parking'])

    # 只有 78 筆有農業用途，且都不具住宅用途，故剔除
    df = df.loc[df['Main_Usage_Farm'] == 0]
    df = df.drop(columns=['Main_Usage_Farm'])

    # NOTICE: 我沒有錢，所以我先只買 6 房以下的
    df = df.loc[df['room'] < 6]

    df = df.loc[df['trading_floors_count'] == 1]

    # 雖然有 95 個樣本包含地下室，但是樣本太少，可能不足以推廣
    # 所以先剔除，剔除完後，都是 0 所以直接 drop
    df = df.loc[df['including_basement'] == 0]
    df = df.drop(columns=['including_basement'])

    # 所有的樣本都不包含人行道，所以直接去除這個 feature
    df = df.drop(columns=['including_arcade'])

    # 剔除交易樓層高度是 -1 (原本有一個樣本)
    df = df.loc[df['min_floors_height'] != -1]

    # 剔除交易建物是 0 個樓層的情況
    df = df.loc[df['building_total_floors'] != 0]

    # 因為車位交易 50 坪以上的資料只有 22 筆，所以先去除
    # 因為浮點數在硬體儲存會有小數點，故不能直接用 == 50.0 去比較
    df = df.loc[df['Parking_Area'] < 49.5]

    # 把農舍，廠辦踢掉
    df = df.loc[df['Building_Types'] < 8]

    # 把超大轉移坪數刪掉
    df = df.loc[df['Transfer_Total_Ping'] < 150]

    # 我先刪除 area_m2, 因為覺得跟 area_ping 的意義很類似，但是不確定會不會有些微差距。
    # 因為在 future data 中，manager 都是 0，所以也把這個欄位刪除
    # trading_floor_count 有 0 的情況，這樣應該不是房屋交易
    df = df.drop(columns=[ 'area_m2', 'manager', 'Building_Material_stone',
                           ]) #'address','TDATE',, '編號','Total_price'

    # Convert the categorical features' dtype to 'category'
    #category_columns = ['Type', 'Month', 'Month_raw',
    #                    'City_Land_Usage', 'Main_Usage_Business',
    #                    'Building_Material_S', 'Building_Material_R', 'Building_Material_C',
    #                    'Building_Material_steel', 'Building_Material_B',
    #                    'Building_Material_W', 'Building_Material_iron',
    #                    'Building_Material_tile', 'Building_Material_clay',
    #                    'Building_Material_RC_reinforce',
    #                    'Parking_Space_Types', 'Building_Types']
    #df.loc[:, category_columns] = df.loc[:,
    #                                     category_columns].astype('category')
    return df


##
y_name = ['Unit_Price_Ping','Total_price']


['Main_Usage_Living']

c_fiture = str_columns.copy()

del_column =['area_m2','manager','including_basement','including_arcade','Main_Usage_Walk','Main_Usage_Living','Main_Usage_Selling','Main_Usage_Manufacturing','Main_Usage_Parking',
 'Main_Usage_SnE','Main_Usage_Farm', 'Building_Material_stone','Non_City_Land_Usage']


c_fiture = pd.Series(c_fiture)[~pd.Series(c_fiture).isin(del_column)].tolist()

data = clean_data(raw_data)
data[str_columns] = data[str_columns].astype('int')
data = data.loc[:,data.columns.isin(str_columns+float_columns+time_columns+ID_columns)]
data = clean_and_drop(data)


test = clean_data(raw_data_test)
test[str_columns] = test[str_columns].astype('int')
test = test.loc[:,test.columns.isin(str_columns+float_columns+time_columns+ID_columns)]
test = clean_and_drop(test)




#raw_data = raw_data[use_column]
#raw_data_test =raw_data_test[use_column]

data[use_column] = data[use_column].astype('str')
test[use_column] = test[use_column].astype('str')

data['combine'] = data[use_column].apply(''.join,axis =1)
test['combine'] = test[use_column].apply(''.join,axis =1)

del_col = ['CPI',
'CPI_rate',
'unemployment rate',
'Pain_index_3month',
'ppen_price',
'high_price',
'low_price',
'close_price',
'qmatch',
'amt_millon',
'return_rate_month',
'Turnover_rate_month',
'outstanding_share_thousand',
'Capitalization_million',
'excess total _ million_usdollars',
'import_price_index_usdollars',
'export_price_index_usdollars',
'export_million_usdollars',
'import_million_usdollars',
'survival_mobility_rate',
'live_deposit_mobility_interest_rate',
'CCI_3month',
'construction_engineering_index',
'Year_Month_Day',
'D_Year_Month_Day',
'transaction_amount',
'Number of successful transactions_TSE',
'Number of transactions_TSE',
'Transaction Amount_OTC',
'Transaction Quantity_OTC',
'Number of transactions_OTC',
'season',
'D_season',
'Real estate market price_income_ratio',]

data= data.loc[:,~data.columns.isin(del_col)]
test= test.loc[:,~test.columns.isin(del_col)]

data.columns = data.columns.str.replace('Note_DistressSale ','Note_DistressSale_')
test.columns = test.columns.str.replace('Note_DistressSale ','Note_DistressSale_')

pd.concat([data,test]).to_csv(adress + '/code/compare_index.csv',index = False)


# a = pd.read_pickle(adress + '/code/Model/model_2022-07-03 19-59-02_Unit_Price_Ping_all_add_variable_notej.pkl')
# d = pd.read_pickle(adress + '/code/Model/model_2022-07-03 19-18-41_Total_price_all_add_variable_notej.pkl')
# b = pd.read_pickle(adress + '/code/Model/model_2022-06-25 17-25-58_Unit_Price_Ping_all_add_variable.pkl')
# e = pd.read_pickle('C:/Users/User/Downloads/model_2022-07-03 19-18-41_Total_price_all_add_variable_notej.pkl')


# In[function test]:
    



def compare_value('Place_id','Type','Transfer_Total_Ping','Building_Types','Month','house_age','min_floors_height')

# 1:14
import datetime as dt
d1 = dt.datetime.now()
difflib.get_close_matches(raw_data_test.query("Place_id=='613'")['combine'].iloc[0], raw_data.query("Place_id=='613'")['combine'])[0]
print(dt.datetime.now()-d1)

# 4:14
d1 = dt.datetime.now()
process.extract(raw_data_test['combine'][0], raw_data['combine'].tolist(),limit=2)
print(dt.datetime.now()-d1)


# 4:30
def minDistance(word1: str, word2: str):
    '編輯距離的計算函數'
    n = len(word1)
    m = len(word2)
    # 有一個字串為空串
    if n * m == 0:
        return n + m
    # DP 陣列
    D = [[0] * (m + 1) for _ in range(n + 1)]
    # 邊界狀態初始化
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    # 計算所有 DP 值
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            left = D[i - 1][j] + 1
            down = D[i][j - 1] + 1
            left_down = D[i - 1][j - 1]
            if word1[i - 1] != word2[j - 1]:
                left_down += 1
            D[i][j] = min(left, down, left_down)
    return D[n][m]

import datetime as dt
d1 = dt.datetime.now()
raw_data['combine'].apply(lambda user: minDistance(user, raw_data_test['combine'][0]))
print(dt.datetime.now()-d1)

# 找不到資料
import re
def fuzzyfinder(user_input, collection):
	suggestions = []
	pattern = '.*?'.join(user_input)	# Converts ‘djm‘ to ‘d.*?j.*?m‘
	regex = re.compile(pattern)		 # Compiles a regex.
	for item in collection:
		match = regex.search(item)	  # Checks if the current item matches the regex.
		if match:
			suggestions.append((len(match.group()), match.start(), item))
	return [x for _, _, x in sorted(suggestions)]

fuzzyfinder(raw_data_test['combine'][0], raw_data['combine'])

# 4:53
d1 = dt.datetime.now()
process.extractOne(raw_data_test['combine'][0], raw_data['combine'])
print(dt.datetime.now()-d1)