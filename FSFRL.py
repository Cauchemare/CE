
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
    return np.sum(df_miss) < (obs*miss_cnt)

def c_variance(df,df_miss,threshold=0.0):
    v=np.var(np.compress(~df_miss,df))
    return  True if v>threshold else False
    
    
    
def c_ensemble(df):
    df_missing=pd.isnull(df)
    return np.all([c_missing(df,df_missing),c_variance(df,df_missing,threshold=0.0)])


class FSFR(BaseEstimator,TransformerMixin):
    """Feature selection from raw_data using lightgbm model
            Args:
                miss_rte:float,0-1;maximum rate of missing allowed
                u_cols:list of columns names;usable columns names;
                type_cols:dict or  mapping type ; variable sets {'cont_cols':list,'ordi_cols':list,'bina_cols':list,'norm_cols':list},
                      keys name:{'cont_cols','ordi_cols','norm_cols','bina_cols'},not necessary to  include all keys name
                      
                y_dist:str,{regression, regression_l1, huber, fair, poisson, quantile, mape, gammma, tweedie, binary, multiclass, 
                            multiclassova, xentropy, xentlambda, lambdarank} default=binary ;distribution of  y  
                            detail see lightgbm document : http://lightgbm.readthedocs.io/en/latest/Parameters.html
                metric:str or set of strs,{mean_absolute_error,mean_squared_error,root_mean_squared_error,mean_absolute_percentage_error,huber,
                            fair,poisson,mean_average_precision,auc,binary_logloss,binary_error,multi_logloss,multi_error};
                            
                  
     Attributes:
        use_cols:list of columns names ; columns satisfy missing condition
        tC_cols:list of columns names ; true continous variables set from  object dtype columns estimated
        tO_cols:list of columns names ; true object variables set from  object dtype columns estimated
                
        gbm:booster;The trained Booster model;see detail http://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api
            used for plotting or predicting.

        
        s_cols:list of columns names;selected columns based on feature_importance of trained booster model
                a rule of thumb:choose feature_importance>2 columns;

    """
    def __init__(self,typ_cols= None,u_cols=None,miss_rate=0.9,y_dist='binary',metric='auc'):
        self.miss_rte=miss_rate
        self.u_cols=u_cols
        self.type_cols=typ_cols
        self.y_dist=y_dist
        self.metric=metric
        
    
    def  fit(self,X,Y):
        return  self._fit(X,Y)
    
    def _fit(self,X,Y):
        global  miss_cnt,obs
        obs=len(X)
        miss_cnt= self.miss_rte
        
        if self.u_cols is not None:
            X=X.reindex(columns=self.u_cols)
        
        
        self.use_cols=X.apply(c_missing)[lambda i:i== True].index


        X_s=X.reindex(columns=self.use_cols)         #retain use_cols
        o_cols=X_s.select_dtypes(include=['object']).columns  #find object dtype columns
        
        self.tC_cols=np.compress(np.isin(o_cols,self.type_cols.get('cont_cols',[])),o_cols) 
        X_s.loc[:,self.tC_cols]=X_s.loc[:,self.tC_cols].astype(np.float16)  #convert continous variables in o_cols to np.float16
        
        self.tO_cols=np.compress(np.isin(o_cols,self.type_cols.get('cont_cols',[]),invert=True),o_cols) #fetch tO_cols,not continuous variables in o_cols

        X_s=pd.get_dummies(X_s,columns=self.tO_cols)
   
        use_cols_f=X_s.apply(c_ensemble)[lambda i:i== True].index
        
        
        X_s=X_s.reindex(columns=use_cols_f)
        
        
        params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': self.y_dist,
        'metric': self.metric,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'histogram_pool_size':1024,
        'is_sparse':True
            }
        
        if self.y_dist in ('binary', 'multiclass', 'multiclassova'):
            Y=Y.values.astype(np.int8).ravel()
            
        features=X_s.columns.tolist()
        X_d_matrix=csr_matrix(X_s.values)
        
        del  X_s
        
        
        train_data=lgb.Dataset(X_d_matrix,Y,feature_name=features,params=params)
        train_data.construct()  
        del  X_d_matrix,Y
        
        self.gbm = lgb.train(params,
                        train_data,
                        num_boost_round=4
                       )

        
        self.s_cols=np.compress(self.gbm.feature_importance()>2,self.gbm.feature_name())   
        
#        print('s_cols corrwith y corr_coef> 0.1',X_s.reindex(columns=self.s_cols).corrwith(y_train).loc[lambda x :x >0.1])
        
        return self
    
    def transform(self,X):
        X_s=X.reindex(columns=self.use_cols)
        X_s.loc[:,self.tC_cols]=X_s.loc[:,self.tC_cols].astype(np.float16)
        X_s=pd.get_dummies(X_s,columns=self.tO_cols)
        
        X_s.rename(columns=partial(re.sub,'\s','_'),inplace=True)   #lightgbm convert implicitlyblank in column name to '_'  connector
        
        
        X_s=X_s.reindex(columns=self.s_cols)
        
        not_finding_columns=self.s_cols[np.all(pd.isnull(X_s),axis=0)]
        
        if len(not_finding_columns)>0:
             print ('X not find columns:',not_finding_columns)        

        return X_s
    

     
        
        


    
    
    
    
