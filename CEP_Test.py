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














