# -*- coding: utf-8 -*-
"""
Created on Sun May 29 13:52:48 2022

@author: Martin

data: clean data
algorithm: lgbm 
times: month
"""

""

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import lightgbm as lgb

from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.model_selection import StratifiedKFold,KFold
from sklearn import metrics

import seaborn as sns
import datetime as dt

import warnings
from sklearn.cluster import KMeans

import random
from bayes_opt import BayesianOptimization
import catboost as cb
import sys
import os




adress = 'C:/Users/User/OneDrive - 銘傳大學 - Ming Chuan University/實價登陸/House_Project/'
sys.path.append(adress)
from eval import simple_evaluate,evaluate_partitions,default_partitions

TRAIN_DATA_PATH = adress + 'output_feature/clean_data_all_add_variable_train.csv'
TEST_DATA_PATH = adress + 'output_feature/clean_data_all_add_variable_test.csv'
CODE_PATH = adress + 'code/MODEL/LGBM_Unitprice_all_add_variable_martin.py'

## 欄位型態確認
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

"""
check 欄位

'Year_Month_Day'
'D_Year_Month_Day'
'transaction_amount'
'Number of successful transactions_TSE'
'Number of transactions_TSE'
'Transaction Amount_OTC'
'Transaction Quantity_OTC'
'Number of transactions_OTC'
'season'
'D_season'
'Real estate market price_income_ratio'
"""

MLFLOW = True
if MLFLOW:

    import mlflow
    SERVER_HOST = os.environ.get('MLFLOW_HOST')
    EXPRIMENT_NAME = 'house_project'
    mlflow.set_tracking_uri(SERVER_HOST)
    mlflow.set_experiment(EXPRIMENT_NAME)
    mlflow.start_run(run_name='Train LGBM allData_add_variable_notej')
    mlflow.log_params({'model_type': lgb.__name__,
                       'training_data': TRAIN_DATA_PATH,
                       'testing_data': TEST_DATA_PATH,
                       'Y_colname':'Unit_Price_Ping',
                       'valid_from_time':'False',
                       #'str_colname':str_columns,
                       #'float_colname':float_columns,
                       #'time_colname':time_columns,
                       #'ID_colname':ID_columns
                       })
    # Log current script code
    mlflow.log_artifact(CODE_PATH)    




raw_data = pd.read_csv(TRAIN_DATA_PATH ,dtype = 'str')
raw_data_test = pd.read_csv(TEST_DATA_PATH ,dtype = 'str')


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
data = clean_and_drop(data)


test = clean_data(raw_data_test)
test[str_columns] = test[str_columns].astype('int')
test = clean_and_drop(test)

#data.to_csv(adress + 'preprocessed_data/20220606/clean_data_train_all.csv',index = False)
#test.to_csv(adress + 'preprocessed_data/20220606/clean_data_test_all.csv',index = False)

data.dtypes.unique()
test.dtypes.unique()


# In[del TEJ]:

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
    
data = data.loc[:,~data.columns.isin(del_col)]
test = test.loc[:,~test.columns.isin(del_col)]

# In[MODEL_train]:
    




Tar = ['Unit_Price_Ping']


params = {  
    'objective': 'regression',
    'learning_rate': 0.01,
    'metric':'rmse',
    'max_depth':7,
    'num_leaves':50
}  


X = data.drop(y_name,1)
y = data[Tar]
X_train,X_valid,Y_train,Y_valid=train_test_split(X,y,test_size=0.2,random_state = 14432) # random_state = seed, stratify 以Y的分佈去拆分X
train = lgb.Dataset(X_train,Y_train, categorical_feature=c_fiture)
valid = lgb.Dataset(X_valid,Y_valid,reference=train, categorical_feature=c_fiture)


gbm_Flag_weight = lgb.train(params,
                train,
                num_boost_round=100000,
                valid_sets=valid,
                early_stopping_rounds=20,verbose_eval=1000)


ypred_final = gbm_Flag_weight.predict(X_valid, num_iteration=gbm_Flag_weight.best_iteration)

# predict_final = predict_threhold(ypred_final,Threhold)
predict_final =ypred_final.copy()
predict_final.max()


# In[set y and predict data]:


sns.distplot(predict_final)
lgb.plot_importance(gbm_Flag_weight)

test_Flag_weight =  pd.DataFrame({'y_predict':predict_final,'Y_valid':Y_valid[Tar[0]]})
print(test_Flag_weight.sort_values(by=['y_predict'],ascending=False).Y_valid.head(int(test_Flag_weight.shape[0]/2)).sum()/test_Flag_weight.Y_valid.sum())

test_Flag_ans = raw_data.loc[test_Flag_weight.index]
test_Flag_ans = test_Flag_ans.merge(test_Flag_weight,left_index=True, right_index = True)


# In[ 各thresholde的指標數]:
def lift_curve( data, y_columns, model_columns, x_tick_num, title):
        '''

        Parameters
        ----------
        data : dataframe
           資料來源
        y_columns : str
            實際值欄位名稱 Ex.AMT, 投資限額
        model_columns : str
            預測值欄位名稱
        x_tick_num : int
            x刻度有幾個
        title : str
            表格title名稱


        '''
        
        a = data[[y_columns, model_columns]]
        model = a.sort_values(model_columns, ascending = False)[str(y_columns)].cumsum().reset_index(drop = True)
        best = a[str(y_columns)].sort_values(ascending = False).cumsum().reset_index(drop = True)
        
        plot_df = (pd.DataFrame([model, best],index=['model','best']).T).reset_index(drop = True)
        plot_df['random'] = np.linspace(0, plot_df.iloc[-1,1],plot_df.shape[0])
        plot_df_rate = plot_df.reset_index().apply(lambda x : x/x.iloc[-1] * 100, axis = 0)    
        plot_df_rate.index = np.array(plot_df_rate['index'])
        plot_df_rate = plot_df_rate.drop('index',axis = 1)
        
        plot_df.plot.line(xticks = np.linspace(0, plot_df.index[-1],x_tick_num),rot = 50, title = title+'_value',grid = True)
        return plot_df_rate.plot.line(xticks = np.linspace(0,plot_df.iloc[-1,1],x_tick_num)/plot_df.iloc[-1,1]*100,rot = 50, title = title+'_rate',grid = True)
        




# Feature Importance
ax = lgb.plot_importance(gbm_Flag_weight, max_num_features=25, height=0.5)
fig = ax.figure
fig.set_size_inches(15, 8)


timestamp = dt.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
FIG_DIR = adress + 'code/Model/'
FIG_name = f'featureimportance_{timestamp}_Unit_Price_Ping_all_add_variable_notej.png'
lift_curve_name = f'liftcurve_{timestamp}_Unit_Price_Ping_all_add_variable_notej.png'
FIG_PATH = f'{FIG_DIR}{FIG_name}'
lift_curve_PATH = f'{FIG_DIR}{lift_curve_name}'
fig.savefig(FIG_PATH,dpi=100, bbox_inches = 'tight')


ax_1 = lift_curve(test_Flag_weight, 'Y_valid', 'y_predict', 20, 'lift_curve')
ax_1.figure.savefig(lift_curve_PATH,dpi=100, bbox_inches = 'tight')
if MLFLOW:
    mlflow.log_artifact(FIG_PATH)
    mlflow.log_artifact(lift_curve_PATH)

# In[Metric]:
# r2, mae, mse = simple_evaluate(gbm_Flag_weight, Y_valid, ypred_final, verbose=True)
r2, mae, mape, mse , scores_mae,scores_mape = evaluate_partitions(gbm_Flag_weight, df = test_Flag_ans, pred_column = 'y_predict', ans_column = 'Y_valid',partitions = default_partitions,index_prefix ='val' )

if MLFLOW:
    mlflow.log_metrics(
        {'val-R-square': r2, 'val-MAE': mae, 'val-MSE': mse, 'val-MAPE': mape})
    mlflow.log_metrics(scores_mae)
    mlflow.log_metrics(scores_mape)

Y_test = test[Tar]
y_pred_test = gbm_Flag_weight.predict(test.drop(y_name,axis =1), num_iteration=gbm_Flag_weight.best_iteration)

test['y_predict'] = y_pred_test
test['y_test'] = Y_test

# r2, mae, mse = simple_evaluate(gbm_Flag_weight, Y_test, y_pred_test, verbose=True)
r2, mae, mape, mse , scores_mae,scores_mape = evaluate_partitions(gbm_Flag_weight, df = test, pred_column = 'y_predict', ans_column = 'y_test',partitions = default_partitions,index_prefix ='test' )
if MLFLOW:
    mlflow.log_metrics(
        {'test-R-square': r2, 'test-MAE': mae, 'test-MSE': mse, 'test-MAPE': mape})
    mlflow.log_metrics(scores_mae)
    mlflow.log_metrics(scores_mape)

import joblib

# Save model
MODEL_DIR = adress + 'code/Model/'
MODEL_NAME = f'model_{timestamp}_Unit_Price_Ping_all_add_variable_notej.pkl'
MODEL_PATH = f'{MODEL_DIR}{MODEL_NAME}'
joblib.dump(gbm_Flag_weight, MODEL_PATH)
if MLFLOW:
    mlflow.log_param('model_path', MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)
    mlflow.lightgbm.autolog(log_input_examples=True, log_model_signatures=True, log_models=True, disable=True, exclusive=True, disable_for_unsupported_versions=True, silent=True, registered_model_name=None)
    mlflow.end_run()

