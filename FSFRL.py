# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:09:34 2018

@author: luyao.li
"""

from  __future__ import absolute_import,print_function,division

import pandas as pd
import numpy as np


from sklearn.base import BaseEstimator,TransformerMixin

import lightgbm as  lgb
import re
from functools import partial
from scipy.sparse  import csr_matrix



def  c_missing(df,df_miss=None):
    if df_miss is None:
        df_miss=pd.isnull(df)
    return df_miss.sum()<(obs*miss_cnt)

def c_variance(df,df_miss,threshold=0.0):
    v=np.var(np.compress(~df_miss,df))
    return  True if v>threshold else False
    
    
    
def c_ensemble(df):
    df_missing=pd.isnull(df)
    return np.all([c_missing(df,df_missing),c_variance(df,df_missing,threshold=0.0)])


class FSFR(BaseEstimator,TransformerMixin):  #FEATURE SELECTION FROM RAW_DATA
    def __init__(self,typ_cols= None,u_cols=None,miss_rate=0.9):
        self.miss_rte=miss_rate
        self.u_cols=u_cols
        self.type_cols=typ_cols
    
    def  fit(self,X,Y):
        return  self._fit(X,Y)
    def _fit(self,X,Y):
        global  miss_cnt,obs
        obs=len(X)
        miss_cnt= self.miss_rte
        
        if self.u_cols is not None:
            X=X.reindex(columns=self.u_cols)
        
        
        self.use_cols=X.apply(c_missing)[lambda i:i== True].index

        #只保留use_cols
        X_s=X.reindex(columns=self.use_cols)
        o_cols=X_s.select_dtypes(include=['object']).columns
        
        
        #将o_cols中 continuous 变量 转换dtype
        self.c_cols=np.compress(np.isin(o_cols,self.type_cols['cont_cols']),o_cols)
        X_s.loc[:,self.c_cols]=X_s.loc[:,self.c_cols].astype(np.float16)
        
        #o_cols 中不属于c_conls 得变量为o2_cols
        self.o2_cols=np.compress(np.isin(o_cols,self.type_cols['cont_cols'],invert=True),o_cols)
        #对o2_cols 进行dummies
        X_s=pd.get_dummies(X_s,columns=self.o2_cols)
   
        self.use_cols2=X_s.apply(c_ensemble)[lambda i:i== True].index
        
        
        X_s=X_s.reindex(columns=self.use_cols2)
        
        
        params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'histogram_pool_size':1024,
        'is_sparse':True
            }
        
        Y=Y.values.astype(np.int8).ravel()
        features=X_s.columns.tolist()
        X_d_matrix=csr_matrix(X_s.values)
        del  X_s
        
        
        train_data=lgb.Dataset(X_d_matrix,Y,feature_name=features,params=params)
#        print('Begin constructing datasets')
        train_data.construct()  #提前优化数据集
        del  X_d_matrix,Y
#        print('Start fitting estimator')
        
        gbm = lgb.train(params,
                        train_data,
                        num_boost_round=4
                       )
        lgb.plot_importance(gbm,max_num_features=10)
        
        self.s_cols=np.compress(gbm.feature_importance()>2,gbm.feature_name())   #2<feature_importance<4.5
        
#        print('s_cols corrwith y corr_coef> 0.1',X_s.reindex(columns=self.s_cols).corrwith(y_train).loc[lambda x :x >0.1])
        
        return self
    def transform(self,X):
        X_s=X.reindex(columns=self.use_cols)
        X_s.loc[:,self.c_cols]=X_s.loc[:,self.c_cols].astype(np.float16)
        X_s=pd.get_dummies(X_s,columns=self.o2_cols)
        X_s.rename(columns=partial(re.sub,'\s','_'),inplace=True)   #lgb.Data默认将所有变量空格填补为_,
        
        
        X_s=X_s.reindex(columns=self.s_cols)
#        print ('X not find columns:',self.s_cols[np.all(pd.isnull(X_s),axis=0)])        

        return X_s
    
    


#FSFR 是否添加u_cols 与最后score没有关系，使用全部变量稍微提高score（17998 rows *2840 --> *5484 cols）  0.6164113344250406--> 0.6166009683706444
    #不过速度更快：80.753601s -> 140.750888s

#数据集越大，score 越高,当训练样本超过3W,score几乎不变，数据越大，s/rows*2840 速度越慢，因为内存不够需要释放空间
# train_size= 0.6  29998 rows -> 23998 rows->17998 rows *2840cols  :0.63447->0.62357->0.616
                      #129.169 s ->  96s --> 65s  0.004s/rows * 2840 cols
                      #39998 rows*2840 cols  269.7252 s train_size=0.8 ->0.6350

     
        
        


    
    
    
    