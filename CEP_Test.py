# -*- coding: utf-8 -*-
"""
Created on Sun May 27 13:02:21 2018

@author: luyao.li
"""

import CEP
from FSFRL import FSFR
import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score


import lightgbm as lgb

import argparse

# =============================================================================
#           Performance
#           ----------
#     #processing time 59.156679s  for data (11999, 5479) containg  fit_transform stage <---3h--4h  180+ faster
#     #transform time 0.418882s    for data (8000, 5479)  just containing  fit stage
#     
#           Usage:
#           ----------
#     #pull the package into foleder: C:\Users\admin\Anaconda3\Lib\site-packages\__pycache__  or set  os.path.chdir(path)
#     
# =============================================================================

parser=argparse.ArgumentParser()
parser.add_argument('n',type=int)
args=parser.parse_args()

cols_names=pd.read_excel(r'D:\CE\CE_TEST_DATA\var_list.xlsx',encoding='utf8',header=None).iloc[:,1].values
           
df=pd.read_csv(r'D:\CE\CE_TEST_DATA\model_sample.csv',encoding='utf8',low_memory=False,nrows=args.n,header=None,names=cols_names)  #q2_pre  more efficient 

type_cols={}
for i in glob.iglob(r'D:\CE\CE_TEST_DATA\*_list.txt'):
    type_cols[os.path.basename(i)[:5]+'cols']=pd.read_csv(i,encoding='utf8',header=None,squeeze=True).values
    
y_col='y'
drop_cols=['cus_num','id','name','cell','resp','other_var1']
drop_cols.append(y_col)
x_cols=[col for col in df.columns  if not col in drop_cols]

nan_y=pd.isnull(df[y_col])

print("input data's shape: {0},drop {1} rows since y-nan".format(df.shape,nan_y.sum()))

df=df.loc[~nan_y,:]


X=df.loc[:,x_cols]
Y=df.loc[:,y_col].astype(np.integer)

del  df,cols_names
#    
#    X=pd.concat([X]*1000,ignore_index=True)
#    y=pd.concat([y]*1000,ignore_index=True)
from collections import defaultdict


scores=defaultdict(list)
times=defaultdict(list)
iter_time=1
sss=StratifiedKFold(n_splits=10)
for train_index,test_index in sss.split(X,Y):
    print('{0} fold'.format(iter_time))
    X_train,X_test=X.iloc[train_index],X.iloc[test_index]
    y_train,y_test=Y.iloc[train_index],Y.iloc[test_index]

    y_train.reset_index(drop=True,inplace=True)
    X_train.reset_index(drop=True,inplace=True)
    
    y_test.reset_index(drop=True,inplace=True)
    X_test.reset_index(drop=True,inplace=True)

    import time
    print('starting  recoded')
    s1=time.time()
    pipeline=make_pipeline(CEP.Recoded(type_cols=type_cols,out=r'C:\Users\admin\Desktop',Order_cats=None),CEP.FSBMD())
    ce_train=pipeline.fit_transform(X_train,y_train) 
    print('fit_transform  X:{0} spends {1:.6f}s'.format(X_train.shape,time.time()-s1))
    times['ce'].append(time.time()-s1)
    
    ce_test=pipeline.transform(X_test)
    
    
    u_cols=[]
    for i in ('cont_cols','bina_cols','ordi_cols','norm_cols'):
        u_cols.extend(type_cols.get(i,[]))
    
    

    print('start lgb')
    s=time.time()       
    a=FSFR(type_cols,u_cols)
    lgb_train=a.fit_transform(X_train,y_train)
    print('lgb processing time {0:.6f}s for  X:{1}*{2}'.format(time.time()-s,len(X_train),len(u_cols)))
    times['lgb'].append(time.time()-s)

    lgb_test=a.transform(X_test)
    
    
    
    combined_train=pd.concat([ce_train,lgb_train],axis=1)
    combined_test=pd.concat([ce_test,lgb_test],axis=1)
    
    #testing stage
    
    
    l=lgb.LGBMClassifier()
    l2 =lgb.LGBMClassifier() 
    l3=lgb.LGBMClassifier() 
    
    print('start fitting')
    l.fit(ce_train,y_train) 
    l2.fit(lgb_train,y_train)
    l3.fit(combined_train,y_train)
    
     
    
    ce_predicted=l.predict_proba(ce_test)[:,1]
    lgb_predicted=l2.predict_proba(lgb_test)[:,1]
    combined_predicted=l3.predict_proba(combined_test)[:,1]
    

    scores['ce'].append(roc_auc_score(y_test,ce_predicted))
    scores['lgb'].append(roc_auc_score(y_test,lgb_predicted))
    scores['combined'].append(roc_auc_score(y_test,combined_predicted))
    iter_time+=1
    
    
print ('ce score:{0:.6f},lgb score:{1:6f},combined score:{2:.6f}'.format(*[np.mean(scores[i]) for i in  ['ce','lgb','combined']] ))
print ('ce takes:{0:.6f}s,lgb spends:{1:6f}s'.format(*[np.mean(times[i]) for i in  ['ce','lgb']] ))


# =============================================================================
# CEP.Recoded:
#     Params:
#         ----------
#
#         type_cols:dict or  mapping type  {'cont_cols':list,'ordi_cols':list,'norm_cols':[],'bina_cols':[]}
#         Profiling: bool  True or False 
#         write_corr:bool corr==spearman
#         order_cats：nestd dict ,e.g:{col_name:[ordered catgories]}
#         
#         
#         y_dist:str {'CLASSES,REGRESSION'}
#         
#         p_lo=0.01
#         p_hi=0.99
#         concrate=0.5  #param for ordinal variables
#         miss_cnt=0.99
#         P=0.05
#         #Profile  params
#         equal_dist='NO'   #'Y'  or 'y'
#         num_category=5
#         
#         impmethodC='mean'  #q_test:scaler.fit()-->ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
#         transf_C=1
#         standardtransform_C=1
#         cap_floor_C=1
#         min_size=10
# 
#         impmethodO='mean'
#         transf_O=1
#         standardtransform_O=1
#         cap_floor_O=None
#         order_cats={'o3':['a','b','c','d']}    
#     
#         #param for nominal or binary varialbes
#         valcount=10
#         minbinn=10
#         bonfer='y'
#         talpha=0.05       # T Test significance level for collapse of bins;
#         nom_method='index'   #binary,binary_encoder,index  binary generate settingwithcopywarning
#         
#         out: path of output (Profiling and correlation)
#         transformer_weights : dict, optional
#         n_jobs=1 
#         random_state=np.random.RandomState(0)
#         write_corr: True
#         write_corr_method:'spearman',optional {'kendall','pearson','spearman'}
# 
# CEP.FSBMD:
#     Params:
#     ----------
#     n_components=n_components
#     y_dist: str  {'classes','regression'}
#     scoring:scorer  function scorer(estimator,y_true,y_test) -->sklearn.metrics
#     n_estimators: n_estimators  for tree_based model
#     rfe:Recursive feature elimination mechanism,str {'RFECV','MYRFECV','SelectFromModel'} -->sklearn.feature_selection.RFECV  SelectFromModel
#     step: int  step
#     cv: cv int sklearn.model_selection
#     transformer_weights:transformer_weights  for decomposition stage and  FSBMD stage
# 
# 
# #get  decomposition  estimator
# destimator = pipeline.named_steps['fsbmd'].get_dtransformer -->sklearn
# 
# #inspect selected_columns for each steps
# pipeline.named_steps['recoded'].selected_columns
# pipeline.named_steps['fsbmd'].selected_columns
# 
# #use  or reset internally estimator of pipeline
# x__train=pipeline.named_steps['recoded'].transform(X_train)
# x___train=pipeline.named_steps['fsbmd'].transform(X_train) --->X_train can't contains str and nan values,object
# 
# 
# 
# enter transformer within  pipeline object
# pipeline.named_steps['recoded']
# pipeline.named_steps['fsbmd']
# enter transformer in featureunion object
# recoded.get_params(deep=True)['re_cont']
# fsbmd.get_params(deep=True)['decomposition']
# 
# 
# #change var attrs for different dtypes
#  're_cont','re_bina'
# set_params(pipeline,dtype)  --->inspect all columns names from  dtype 
# set_params(estimator,dtype,var) ---> inspect all attrs of column    
# set_params(estimator,dtype,var,attr)    --> inspect attribute value of attr
# set_params(estimator,dtype,var,attr,value)    --> set value of var 's attr
# 
# then reload transform method
# x_reloaded=pipeline.transform(x)
# 
# #save model
# 
# import  pickle
# pickle.dump(pipeline,open(r'C:\Users\admin\Desktop\pipeline.pkl','wb'))
# pipeline=pickle.load(open(r'C:\Users\admin\Desktop\pipeline.pkl','rb'))
# 
# binary_file=pickle.dumps(pipeline)
# pipeline=pickle.loads(binary_file)
# =============================================================================














