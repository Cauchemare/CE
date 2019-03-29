# -*- coding: utf-8 -*-


"""
Library for selecting features

According to different dtype,we generate recoding rules for different variables,and retain variables filtered by set of conditions. 

more detail see https://github.com/Cauchemare/CE


"""
from  __future__ import absolute_import,print_function

from sklearn.base import BaseEstimator,TransformerMixin,MetaEstimatorMixin
from sklearn.base import clone
from sklearn.utils import check_X_y
import numpy as np
import pandas as pd
from statistics import median

from collections import namedtuple


from sklearn.feature_selection import f_regression,f_classif
from sklearn.linear_model import   LinearRegression,LogisticRegression 
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,roc_auc_score,make_scorer
from sklearn.feature_selection import RFE,RFECV
from sklearn.pipeline import FeatureUnion,make_pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.linear_model import RandomizedLogisticRegression,RandomizedLasso


import warnings
from scipy.stats import  t
import category_encoders as ce
from pandas.api.types import CategoricalDtype 
from  itertools import combinations
from operator import attrgetter
import os


warnings.filterwarnings("ignore",category =RuntimeWarning)
warnings.filterwarnings("ignore",category =DeprecationWarning)

'''
Feature  selection based on models FSBM
'''

class  FSBM(BaseEstimator,TransformerMixin):
    """Feature selection based on models,contains two type models(linear-model and tree_based model)
        Args:
            linear_esimator:estimator instance,randomiedlasso or randomizedlogisiticregression see sklearn.randomiedlasso or sklearn.randomizedlogisiticregression
            tree_estimator:estimator instane,transformer embedding tree_based model for selecting features based on importance weights,see sklearn.RFECV
    
        Attributes:
            ls:an estimator instance,see linear_esimator
            te:an estimator instance,{'RFECE','myrfe','selectfrommodel'} instance,an estimator,see tree_estimator
            choosed_features:str,union of features selected by  linear_esimator and tree_estimator
            cols:str,columns names
    Return:
        self
    """
    def __init__(self,linear_estimator,tree_estimator):
        self.ls=linear_estimator   #randomiedlasso or randomizedlogisiticregression
        self.te=tree_estimator     #rfe,myrfe,selectfrommodel containing tree_based classifier or regression
        

    def fit(self,X,y):
    
        return self._fit(X,y)

    def _fit(self,X,y):
        #X:dataframe 
        try:
            cols=X.columns
        except  AttributeError: 
            cols=np.array(['fsbm'+'%d'%i for i in range(X.shape[1])])
            X=pd.DataFrame(X,columns=cols)
        for i in vars(self).keys():
            attrgetter(i)(self).fit(X,y)
        self.choosed_features=set(cols[self.ls.get_support()]).union(cols[self.te.get_support()])
        self.cols=cols
        return self
    
    def transform(self,X):
        if isinstance(X,np.ndarray):
            X=pd.DataFrame(X,columns=self.cols)
        return X.loc[:,self.choosed_features]
    

def  auc_mc(y_t,y_p):   
    #compute auc_score,compatible with multiclass y
    classes=np.arange(len(np.unique(y_t)))
    
    y_predict=label_binarize(y_p,classes=classes)
    y_true=label_binarize(y_t,classes=classes)
    try:
       s= roc_auc_score(y_true.ravel(),y_predict.ravel())
    except Exception:
        print('auc_mc error',y_true[:10],y_predict[:10])
        return 
    
    return s


class MyRFECV(RFE,MetaEstimatorMixin):
    """Based on greedy algorithms,from top to base,select 'step' variables to delete for each round until no score increase
        Args:
            X:array_like,not DataFrame,not containing np.nan or np.infinite,
                all object columns are forced to convert to float ,raise since failure
            y:array_like,target data
            estimator:estimator instance,an estimator to fit selected_columns sub-data
            n_feature_to_select:int or None (default=None),The number of features to select. If None, half of the features are selected.
            step:int or float, optional (default=1)
            verbose：int, default=0,Controls verbosity of output.
            cv:int, cross-validation generator or an iterable, optional
                    Determines the cross-validation splitting strategy. Possible inputs for cv are:
        
                    None, to use the default 3-fold cross-validation,
                    integer, to specify the number of folds.
                    An object to be used as a cross-validation generator.
                    An iterable yielding train/test splits.
                    For integer/None inputs, if y is binary or multiclass, sklearn.model_selection.StratifiedKFold is used. If the estimator is a classifier or if y is neither binary nor multiclass, sklearn.model_selection.KFold is used.
        
                    Refer User Guide for the various cross-validation strategies that can be used here.
            scorer:string, callable or None, optional, default: None
                A string (see model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y).
        Attrs:
            see  sklearn.feature_selection.RFECV
    
    
    """
    def __init__(self, estimator, n_features_to_select=None, step=1,
                 verbose=0,cv=3,scorer=accuracy_score):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
        
        self.cv=cv
        self.scorer=None
    
    def  _fit(self,X,y,scorer=None):
        X,y=check_X_y(X,y,True)
        n_features= X.shape[1]
        
        if self.n_features_to_select is None:
            n_features_to_select=n_features//2
        else: 
            n_features_to_select=self.n_features_to_select
            
        
        if 0.0 < self.step <  1.0:
            step=int(max(1,self.step*n_features))
        else:
            step=int(self.step)
        if step<0:
            raise ValueError("Step must be >0")
            
        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)
        f=np.arange(n_features)
        
        
        
        #Elimination by metrics
        while np.sum(support_)>  n_features_to_select:
            features=f[support_]
            estimator=clone(self.estimator)
            
            if self.verbose > 0:
                print("Fitting estimator with %d features." % np.sum(support_))
                
            max_score=0
            max_support=None
            dict_max_scores={}
            j=1
            threshold=min(step,np.sum(support_)-n_features_to_select)
            for i in combinations(features,threshold):
                s=support_.copy()
                s[list(i)]=False
                
                score=cross_val_score(estimator,X[:,f[s]], y,scoring=self.scorer,cv=self.cv).mean()
                if score>=max_score:
                    max_score=score
                    max_support=s
        
            if len(dict_max_scores)==0: 
                    dict_max_scores[j]=max_score
                    j+=1
                    support_=max_support
                    ranking_[np.logical_not(support_)]+=1
                    
            else:
                if max_score>=dict_max_scores[j-1]:
                    dict_max_scores[j]=max_score
                    j+=1
                    support_=max_support
                    ranking_[np.logical_not(support_)]+=1
                else:
                    features=f[support_]
                    self.estimator_=clone(self.estimator)
                    self.estimator_.fit(X[:,features],y)
                    self.n_features_ =support_.sum()
                    self.support_ =support_
                    self.ranking_=ranking_
                    return self
        features=f[support_]
        self.estimator_=clone(self.estimator)
        self.estimator_.fit(X[:,features],y)
        self.n_features_ =support_.sum()
        self.support_ =support_
        self.ranking=ranking_
        return self




def writecorr(X,X_selected,y,out=None,method='spearman'):
    """output write-correlation  for two raw_data 
        Args:
            X:DataFrame,array_like
            X_selected:DataFrame,array_like
            y:DataFrame,array_like,target data
            method:{‘pearson’, ‘kendall’, ‘spearman’},
                pearson : standard correlation coefficient
                kendall : Kendall Tau correlation coefficient
                spearman : Spearman rank correlation
            out:str of filepath
        Return:
            None
    
    """

    writer=pd.ExcelWriter(out)
    
    
    X.apply(lambda col: col.corr(y,method=method),axis=0).to_excel(writer,'all_variables',encoding='utf8',header=['y_target'])
    X_selected.apply(lambda col: col.corr(y,method=method),axis=0).to_excel(writer,'seleted_variables',encoding='utf8',header=['y_target'])
    
    writer.save()
    
    return  
    


# =============================================================================
# """
# Set of conditions for multiply type vars
#     c_missing:missing_count bounded
#     c_uniform:low_percentile and high_percentile of no-missing_data inequal
#     c_concentrate:the most_freq type (including Nan type) constraint
#     c_concentrate_bina:1 count 's constaint
#     
#     c_dep:minimum_y != maximum_y for no_missing data
#     c_numcat: number of categories constraint
# """"
# =============================================================================
def  c_missing(df,df_missing=None):
    return df_missing.sum()<(obs*miss_cnt)

def c_uniform(df,df_missing=None,p1=0.01,p2=0.99):
    df_nomiss=np.compress(~df_missing,df).astype(np.float32)
    if df_nomiss.empty:
        return  False
    return np.percentile(df_nomiss,p_lo)!=np.percentile(df_nomiss,p_hi)

def c_concentrate(df):
    return  obs*concrate > df.value_counts(dropna=False).iat[0]


def  c_concentrate_bina(df):
    cnt_ck=np.sum(i in  [1,'1','y','Y'] for i in df)  
    return   obs*concrate >=cnt_ck or  cnt_ck >=obs*(1-concrate)
    


def c_dep(df,df_missing=None,y_col=None):    
    if isinstance(y_col,np.ndarray):
        y_col=pd.Series(y_col,name='y')
    df_y=y_col.loc[~df_missing]   
    if df_y.empty:
        return False
    return  max(df_y)!=min(df_y)  

def c_numcat(df,df_missing=None):
    return  valcount!=0 and valcount>len(set(df.loc[~df_missing]))
    
    

#set of constraints for continuous  var
def c_cont(df):
    df_miss=pd.isnull(df)
    return all([c_missing(df,df_miss), c_uniform(df,df_miss,p_lo,p_hi),c_dep(df,df_miss,y)])

#set of constraints for ordinal  var
def c_ordi(df):
    df_miss=pd.isnull(df)
    return all([c_missing(df,df_miss),c_concentrate(df),c_dep(df,df_miss,y)])

#set of constraints for norminal  var
def c_norm(df):
    df_miss=pd.isnull(df)
    return all([c_missing(df,df_miss),c_concentrate(df),c_numcat(df,df_miss)])

#set of constraints for binary  var
def c_bina(df):
    df_miss=pd.isnull(df)
    return  all([c_missing(df,df_miss),c_concentrate_bina(df)])

'''
Profile programmes :various and  ordinal vars
'''


def prof1(X,y,var_name):
    """Profiling especially for continuouse or ordinal variable
    return  DataFrame  |var_name | count | Average_DV | category |
    
    var_name :Categories name for var_name,'-1' means missing categorie
    count：number count for Categories
    Average_DV:mean of target data for Categories
    category:Categories name,'missing' mean missing categorie
                            
    """
    
    df=pd.DataFrame()
    df.loc[:,var_name]=X.loc[:,var_name] 
    df['y_dep']=y
    df.fillna({var_name:'-1'},inplace=True)
    prof=df.groupby(var_name).agg([('count',len),('Average_DV',np.mean)])
    
    prof.columns = prof.columns.droplevel(0) 
    prof.reset_index(inplace=True)
    
    prof['category']=np.where(prof.loc[:,var_name]=='-1','missing',prof[var_name])
    return  prof

def prof2(X,y,var_name):
    """Profiling especially for continuouse or ordinal variable
    Cut data between p1 and p99 into num_category parts
    Create DataFrame's structure like: | bin | count | lo | hi | Average_DV | category |
    
    """
    #bin==-1:missing value
    temp=pd.DataFrame()   
    temp.loc[:,var_name]=X.loc[:,var_name]
    if equal_dist.lower()=='yes':
        p1 =np.nanpercentile(temp,1)   
        p99=np.nanpercentile(temp,99)
        _,bi=pd.cut([p1,p99],bins=num_category,retbins =True)
        temp.loc[:,'bin']=pd.cut(temp.fillna(-1).values.ravel(),bins=bi,labels=np.arange(1,num_category+1))  
        
        temp.loc[:,'bin']=np.where(temp[var_name]<p1,1,
                                  np.where(temp[var_name]>p99,num_category,
                                          np.where(pd.isnull(temp[var_name]),-1,temp.bin)))   
    else:    
        temp.loc[:,'bin']=pd.qcut(temp.values.ravel(),num_category,labels=range(1,num_category+1),precision=8,duplicates='drop')  
        temp.loc[:,'bin' ]=np.where(temp.bin.isnull(),-1,temp.bin)
        
    temp['y_dep']=y
    prof=temp.groupby('bin').agg({var_name:[('count',len),('lo',min),('hi',max)],'y_dep':[('Average_DV',np.mean)]}) 
    prof.columns=prof.columns.droplevel(0)
    prof.reset_index(inplace=True)
    prof.sort_values('bin',inplace=True)
    
    prof['hi_str']=prof['hi'].astype(str)    
    prof['lo_str']=prof['lo'].astype(str)

    
    prof['category']=np.where(prof.bin==-1,'missing',
                                np.where(prof.bin==1,'low to '+prof.hi_str,
                                         np.where(prof.bin==prof.bin.iat[-1],prof.lo_str+' to high',
                                                  prof.lo_str+' to '+prof.hi_str)))
    del prof['hi_str'],prof['lo_str']
    return prof




def prof3(typ,prof,var_name,label,ct,mean_y=None):  
    """Profiling especially for continuouse or ordinal variable
    prof1->prof2->prof3
    
    Create DataFrame's structure like: | category | count | percent | Average_DV | index | star | variable | label | type |
    
    Args:
        ct:int,total number for var_name
        mean_y=mean of target data
    
    """
    prof3=pd.DataFrame(columns=profile_cols[3:])
    prof3.loc[0,:]=['overall',ct,1,mean_y,100,'']
    
    #recreate prof  table
    prof2=prof.loc[:,['category','count']]
    prof2['percent']=prof2.loc[:,'count']/ct   
    prof2['Average_DV']=prof.loc[:,'Average_DV']
    prof2['index']=(prof2.loc[:,'Average_DV']/mean_y)*100
    prof2['star']=np.where(prof2.index >=110,'* (+)',
                             np.where(prof2.index >100,' (+)',
                                      np.where(prof2.index <=100,'* (-)',
                                               np.where((prof2.index<=90)&(prof2.index>=50),' (-)',' (0)'))))
    prof3=pd.concat([prof3,prof2],ignore_index=True)
    prof3['variable']=var_name
    prof3['label']=label
    prof3['type']=typ
    return prof3




def prof4(tmp5,var):
    """Profiling especially for normally variable
    Create DataFrame's structure:|fgroup | category | count |  Average_DV |
    
    
    """
    tmp=tmp5.loc[:,['xcount','xmean',var,'fgroup']].rename(columns={'xcount':'count','xmean':'Average_DV'})
    tmp=tmp.groupby('fgroup').agg({var:[('category',reducestr)],'count':[('count',reducescore)],'Average_DV':[('Average_DV',reducescore)]})
    tmp.columns =tmp.columns.droplevel(0)
    return tmp

def prof5(X,y,var):
    """Profiling especially for binary variable
    
    """
    s=X.loc[:,var]
    typ=s.dtypes.name
    
    df=pd.DataFrame()
    df.loc[:,'y_dep']=y
    df.loc[:,'new_var']=np.where([i in [1,'1','y','Y']  for i in s],1,0)
    prof=df.groupby('new_var').agg([('count',len),('Average_DV',np.mean)])
    prof.columns=prof.columns.droplevel(0)
    
    if typ=='object':
        prof['category']=np.where(prof.index==0,'missing,0,N','1,Y')
    else:

        prof['category']=np.where(prof.index==0,'missing,0','1')
    return prof






def  transform_cont(d,_trans=None):
    """Transformer data in_place
        Args:
            d:series or array_like,need_transfored data
            trans:{'SQ_','SR_','LN_'},abbreviation of trans method name
    """

    if  _trans=='SQ_':
        d.loc[:]=d**2
    elif _trans=='SR_':
        d.loc[:]=np.ma.sqrt(np.where(d<0,0,d)).data
    elif _trans=='LN_':
        d.loc[:]=np.ma.log(np.where(d<0.00001,0.000001,d)).data
    return d

def nonmissing(var,var_lb,var_ub,transformation='Y',clip='y'):
    '''Create DataFrame containing each var_data and its transformation,aiming to select the best transformation  in the next step
        var:array_like or series_like values,input data containing  y column(y,var1)
        var_lb:array_like,minimum value
        var_ub:arrray_like,maxinmum value
        transfmation:boolean,if we  implement transfomation for meta_data
    
    Return:
        np.ndarray object,  (var1,var2,...,y)
    '''
    if not isinstance(var,np.ndarray):
        var=np.array(var)
    assert var.shape[1]==2,AttributeError('creating nonmissing data,input data must be 2 columns')
    var=np.compress(~pd.isnull(var[:,0]),var,axis=0)
    y=var[:,0][:,np.newaxis]  #guarantee y'shape [n,1]
    var=var[:,1][:,np.newaxis] #guarantee var'shape [n,1]
    var=np.clip(var,var_lb,var_ub) 
    
    v=var.copy()  #store meta_data
    
    if transformation:
        var=np.append(var,v**2,axis=1)
        var=np.append(var,np.ma.sqrt(np.where(v<0,0,v)).data,axis=1)  
        var=np.append(var,np.ma.log(np.where(v<0.00001,0.000001,v)).data,axis=1)  
    var=np.append(var,y,axis=1)
    return  var

def  pnum(X,y=None,cont_var= None,impmethod=None,transf=None,standardtransform=None,cap_floor=None):
        """Main estimator for continuous and ordinal vars
            Args:
                X:DataFrame
                y:series or array_like,target data
                cont_var:str,column name considered as continuous vars
                impmethod:str,method of imputation
                transf:boolean,if we transformer data
                standardtransform:boolean,if we standardtransform data
                cap_floor:boolean, if we clip input data 
        Return:
            dict,recoded method for  continuous variable
 
        
        """
        df=X.loc[:,cont_var]
        df_miss= pd.isnull(df)
        
        nmiss=df_miss.sum()
        des=df.describe(percentiles=[0.01,0.25,0.75,0.99]).values  
        var=namedtuple('describe',['count', 'mean', 'std', 'min', 'p1', 'p25', 'p50', 'p75', 'p99', 'max'])._make(des)
        med=median(df[~df_miss]) 
        
        iqr=np.max((var.p75-var.p25,var.p99-var.p75,var.p25-var.p1))
        var_lb=min(max(var.p25-1.5*iqr,var.min),var.p1)
        var_ub=max(min(var.p75+1.5*iqr,var.max),var.p99)
        if var_lb==var_ub:
            var_lb=var.min
            var_ub=var.max
        var_mid=(var.max-var.min)/2
        
        assert isinstance(impmethod,str),AttributeError('impmethod must be a string')
        impmethod=impmethod.lower()
        if impmethod  in ('mean','std'):
            var_miss=var.mean
        elif impmethod in ('median', 'iqr','mad'):
            var_miss=med
        elif  impmethod in ('range'):
            var_miss=var.min
        elif  impmethod in ('midrange'):
            var_miss=var_mid
        elif  impmethod in  ('sum','euclen','ustd','maxabs'):
            var_miss=0   
            
        df_nonmissing=nonmissing(np.append(y[:,np.newaxis],df.values[:,np.newaxis],axis=1),var_lb,var_ub,transformation=transf)   
        
        var_names=[cont_var]
        if transf:
            var_names.extend([i+cont_var for  i in trans_prefix])
        
        ck=nmiss<min_size
        
        global y_dist  
        y_dist=y_dist.lower()   
        if  ck==False and impmethod =='er': 
            missing_rr=np.mean(np.compress(df_miss,y,axis=None)) 
            if  y_dist=='classes':
                    if missing_rr==min_y :  
                        missing_rr=min_y+0.0001
                    elif missing_rr==max_y:
                        missing_rr=max_y-0.0001

        score_func=f_classif if y_dist=='classes' else f_regression
        estimator_ref=LogisticRegression() if y_dist=='classes' else LinearRegression()
        #ATTENTION!!
        df_nonmissing_x=np.compress(~pd.isnull(df_nonmissing[:,0]),df_nonmissing,axis=0)
        #find optimal variable
        y_nomissing_x=df_nonmissing_x[:,-1]
        supports=RFE(estimator_ref,n_features_to_select= 1,step=1).fit(df_nonmissing_x[:,:-1],y_nomissing_x).support_  
        new_var=np.compress(supports,var_names,axis=None)[0]
        p_value=np.compress(supports,score_func(df_nonmissing_x[:,:-1],y_nomissing_x)[1])[0]
        
        #estimate whether y is multiclass
        if  len(set(y_nomissing_x))>2:
            y_missing_x_b=np.where(y_nomissing_x==1,1,0)
        else:
            y_missing_x_b=y_nomissing_x
        est=estimator_ref.fit(np.compress(supports,df_nonmissing_x,axis=1),y_missing_x_b)   
        _trans=new_var[:3] if new_var[:3] in trans_prefix else ''
        
        intercept=est.intercept_[0]
        coef=est.coef_[0][0]
   
        
        if impmethod=='er':
            if  y_dist=='classes':
                if ck is True or p_value>0.05:  
                    miss_impute=med
                else:
                    if _trans=='SQ_':
                        miss_impute=np.sqrt(max((np.log(missing_rr/(max_y-missing_rr))-intercept)/coef,0))
                    elif _trans=='SR_':
                        miss_impute=(np.log(missing_rr/(max_y-missing_rr))-intercept)/coef**2
                    elif _trans=='LN_':
                        miss_impute=np.exp((np.log(missing_rr/(max_y-missing_rr))-intercept)/coef)
                    elif _trans=='IV_':
                        miss_impute=-1/(np.log(missing_rr/(max_y-missing_rr))-intercept)/coef
                    elif  _trans=='EP_':
                        miss_impute=-np.log(max(-(np.log(missing_rr/(max_y-missing_rr))-intercept)/coef,0.0001))
                    else:
                        miss_impute=(np.log(missing_rr/(max_y-missing_rr))-intercept)/coef 
            else:
                if ck==True or p_value>0.05:
                    miss_impute=med
                else:
                    if _trans=='SQ_':
                        miss_impute=np.sqrt(max((missing_rr-intercept)/coef,0))
                    elif _trans=='SR_':
                        miss_impute=(missing_rr-intercept)/coef**2
                    elif _trans=='LN_':
                        miss_impute=np.exp((missing_rr-intercept)/coef)
                    elif _trans=='IV_':
                        miss_impute=-1/((missing_rr-intercept)/coef)
                    elif  _trans=='EP_':
                        miss_impute=-np.log(max(-(missing_rr-intercept)/coef,0.0001))
                    else:
                        miss_impute=(missing_rr-intercept)/est.coef#q_test:
            var_miss=np.clip(miss_impute,var_lb,var_ub)
 
        if  p_value<=P:
            d=df.copy()
            if standardtransform:
                d.fillna(var_miss,inplace=True)
                if cap_floor:
                    d=np.clip(d,var_lb,var_ub)
                d=transform_cont(d,_trans)
                scaler=StandardScaler()
                scaler.fit(d.values.reshape(-1,1))  
            else:
                scaler =None   
        else:
            new_var=''
            scaler=None
#           print  ('drop variable {0} since its p_value:{1:.3f} greater than p-expected'.format(cont_var,p_value))    -->format(cont_var,) 
        
        return  {cont_var:{'var_miss':var_miss,'var_lb':var_lb,'var_ub':var_ub,'new_var':new_var,'scale':scaler}}
 


def creatv(tmp6,var,first=None):
    """Updata table  tmp6,adding two columns,k-->index for the same value of var
                                             lk --> maximum index for the same value of var    
    """
    for i in range(len(tmp6)):
            va=tmp6.at[i,var]
            if i ==0:
                tmp6.loc[i,'v']=str(va)
                tmp6.loc[i,'k']=1   #k  first.diff
                
            else:
                d=tmp6.at[i,first]
                d1=tmp6.at[i-1,first]
                v1=tmp6.at[i-1,'v']
                if d1==d:
                    tmp6.loc[i,'v']=v1+','+str(va)
                else:   
                    tmp6.loc[i,'v']=str(va)
                    tmp6.loc[i-1,'lk']=1   # lk   last.diff
                    tmp6.loc[i,'k']=1
    tmp6.loc[i,'lk']=1
    return  tmp6


 
#Profiling functions
def  reducestr(series):
    return  ','.join(series.values)

def reducescore(series):
    return  series.values[-1]   

def pnom(X,y,var,nom_method):
    """Main estimator for norminal variable,only valid for nom_method in   {'index','binary'}
        Args
        ----------
        X:DataFrame,train_X,imputed with 'missing' X str Array containing only self.use_cols
        y :array or series,train_y,
        var:str,name of var
    
    Result
    ----------
    dict,{var_name: new_var_name if nom_method  == 'binary'
                  : value if now_method ==  'index' }  contains  missing categorie
    
    """
    df=pd.DataFrame()
    df.loc[ :,var]=X.loc[:,var]
    
    df['y_dep']=y
    
    
    tmp=df.groupby(var).agg([('dcount',len),('dmean',np.mean),('dvar',np.var)])

    tmp.columns=tmp.columns.droplevel(0)
    tmp.reset_index(inplace=True)     
    tmp.sort_values('dmean',ascending=False,inplace=True)
    tmp.index =np.arange(len(tmp))
    
    #tmp1:cumulated table
    cols1=['tcount','tmean','tvar','tgroup','tgrpcnt']
    tmp1=pd.DataFrame(columns=cols1)
    tmp1=pd.concat([tmp1,tmp],axis=1)
    
    tmp1.loc[:,'cumcnt'] =np.cumsum(tmp1.dcount)    
   
    cum=tmp1.at[len(tmp1)-1,'cumcnt']
    for i in range(len(tmp1)):
        if i ==0:
            tmp1.loc[i,['tcount','tmean','tvar']]=tmp1.loc[i,['dcount','dmean','dvar']].values
            tmp1.loc[i,['tgroup','tgrpcnt']]= [1,1]
        else:
            tc,tm,tv,tg,tgc=tmp1.loc[i-1,cols1].values
            dc,dm,dv=tmp1.loc[i,['dcount','dmean','dvar']].values
            if  (tc <= minbinn)  or ((cum-tc)>=(obs-minbinn)):    
                tmp1.at[i,'tmean']=((tc*tm)+(dc*dm))/(tc+dc)
                tmp1.at[i,'tvar']=((tc-1)*tv+(dc-1)*dv)/(tc+dc-2)
                tmp1.at[i,'tcount']=tc+dc
                tmp1.at[i,'tgroup']=tg
                tmp1.at[i,'tgrpcnt']=tgc+1
            else:
                tmp1.loc[i,cols1]=[dc,dm,dv,tg+1,1]     


    #tmp2=tmp1.assign(rn=tmp1.sort_values('tgrpcnt,ascending=False').groupby('tgroup').cumcount()+1).query('rn==1')        
    tmp2=tmp1.assign(rn=tmp1.sort_values('tgrpcnt',ascending=False).groupby('tgroup').cumcount()+1).query('rn==1').loc[:,cols1]
    tmp2.index=np.arange(len(tmp2))
    #calculate bonferroni  adjustement on alpha  value
    if bonfer.lower()=='yes':
        ncomps=len(tmp2.tgroup.unique())-1
        if ncomps>1:
            ftalpha=talpha/ncomps
        else:
            ftalpha=talpha
    #Collapse values based on variance:
        #tmp3 cumulated table
    cols3=['fcount','fmean','fvar','fgroup','grpcnt']
    tmp3=pd.DataFrame(columns=cols3)
    tmp3=pd.concat([tmp2,tmp3],axis=1)

    for  i in range(len(tmp3)):
        if i==0:
            tmp3.loc[i,['fcount','fmean','fvar']]=tmp3.loc[i,['tcount','tmean','tvar']].values   
            tmp3.loc[i,['fgroup','grpcnt']]= [1,tmp3.at[i,'tgrpcnt']]  
        else:
            fc,fm,fv,fg,fgc=tmp3.loc[i-1,cols3].values
            tc,tm,tv,tg,tgc=tmp3.loc[i,cols1].values
            pvar=((fc-1)*fv+(tc-1)*tv)/(fc+tc-2)
            t_value=(fm-tm)/np.sqrt(pvar*(1/fc+1/tc))
            df=(fc+tc-2)
            prob=(1-t.pdf(abs(t_value),df))   #return  proba x from  t-standard df  degre of freedm  T distribution
            if prob  <=ftalpha:
                tmp3.loc[i,cols3]=[tc,tm,tv,fg+1,tgc]
            else:
                tmp3.at[i,'fmean']=((fc*fm)+(tc*tm))/(fc+tc)
                tmp3.at[i,'fvar']=((fc-1)*fv+(tc-1)*tv)/(fc+tc-2)
                tmp3.at[i,'fcount']=fc+tc
                tmp3.at[i,'fgroup']=fg
                tmp3.at[i,'grpcnt']=fgc+tgc
    tmp3=tmp3.reindex(columns=['tgroup','fgroup','fcount','fmean','grpcnt'])
    tmp4=tmp3.assign(rn=tmp3.sort_values('grpcnt',ascending=False).groupby('fgroup').cumcount()+1).query('rn==1')   
    tmp4.index=np.arange(len(tmp4))
    tmp4=tmp4.reindex(columns =['fgroup','fmean','fcount','grpcnt']).rename(columns={'fmean':'xmean','fcount':'xcount','grpcnt':'group_size'})
    
    tmp4=pd.merge(tmp3.reindex(columns=['tgroup','fgroup']),tmp4,on='fgroup')
    tmp4['Index']=(tmp4.loc[:,'xmean']/average_all)*100
    tmp4['diff']=np.abs(tmp4.xmean-average_all)
    tmp4['name']=var
    
    
    #sort to keep  smaller groups first
    tmp4.sort_values(['group_size','diff'],ascending=[True,False],inplace=True)
    last_group=tmp4.at[len(tmp4)-1,'fgroup']
    
    tmp5=pd.merge(tmp1.reindex(columns =[var,'tgroup','dmean','dcount']),tmp4.reindex(columns=['tgroup','name','fgroup','xmean','xcount','diff','Index','group_size']),on='tgroup')
    del tmp5['tgroup']
    
    nom_method=nom_method.lower()
    #write out  recode code
    if  nom_method=='binary':
        tmp6=tmp5.loc[tmp5.fgroup!=last_group,:]    
        if tmp6.empty:
            tmp6=tmp5
        tmp6.sort_values(['group_size','diff'],ascending=[True,False],inplace=True)
        tmp6 =creatv(tmp6,var,first='diff').query('lk==1')  #not matter var's dtype,convert it to str dtype
        
        tmp6.loc[:,'new_var']=(myprefix+tmp6.loc[:,'name']+'_X'+tmp6.fgroup.astype(str)).values   
        mapping_list=dict()
        for vs,nv in  tmp6.loc[:,['v', 'new_var']].itertuples(index=False):
            mapping_list.update(dict.fromkeys(vs.split(','),nv))
        return  tmp5,{var:mapping_list}

    else:
        if  nom_method=='index':
            tmp5.loc[:,'val']=tmp5.index
        else:
            tmp5.loc[:,'val']=tmp5.xmean
        return   tmp5,{var:dict(zip(tmp5[var],tmp5['val']))}       
  
  
#CE2_RECOD and profile architecture
'''
FeatureUnion([('re_cont',re_cont(continous_variables)),('re_ordi',re_ordi(ordinal_variables))
             ,('re_norm',re_norm(normina_varalbe)),('re_bina',re_bina(binary_variables))])

while generating Profile file.
''' 
from sklearn.externals.joblib import Parallel,delayed

def transform_one(trans,weight,X):
    result=trans.transform(X)
    if not isinstance(result,pd.DataFrame):
        result=pd.DataFrame(result,columns=[myprefix+'{}'.format(i) for i in range(result.shape[1])])
    if weight is None:
        return result
    return result*weight



class MyFeatureUnion(FeatureUnion):   
    """See sklearn.pipeline.FeatureUnion,the only change lies in the type of output:np.array -->pandas.DataFrame
    """
    def transform(self,X):
        Xs = Parallel(n_jobs=self.n_jobs)(
        delayed(transform_one)(trans, weight, X)
        for name, trans, weight in self._iter())
        if not Xs: #if there exits Non-Dataframe in Xs,All transformers are None
            return pd.DataFrame()
        Xs=pd.concat(Xs,axis=1)
        return Xs


class  re_cont(BaseEstimator,TransformerMixin):
    """Transformer for continuous varibles,generate recoded rule for each variable and output Profiling file and correlation_with_y fil
            Args:
                cont_variables:sequence,continous variables
                standardtransform:boolean,if standardtransform is True,default True
                out:str,path for output Profiling file
                
        Return:
            fit():instance,self
            transform():DataFrame,result dataframe
            
        Methods:
            see sklearn.base.TransformerMixin or  simply  sklearn.linear_model.LogisticRegression
    """
    def __init__(self,cont_variables,out):
        self.vars=cont_variables
        self.standardtransform=standardtransform_C
        self.out=out
        
    def fit(self,X,y=None): 
       
        X=X.loc[: ,self.vars].astype(np.float32)  #except var.dtype is 'object'
        
        self.use_cols=X.astype(np.float32).apply(c_cont)[lambda i: i== True].index
        self.fit={}
        X_used=X.reindex(columns=self.use_cols)  #subset X_train data
        
        for i in self.use_cols:
            d=pnum(X_used,y,cont_var= i,impmethod=impmethodC,transf=transf_C,standardtransform=standardtransform_C,cap_floor=cap_floor_C)
            self.fit.update(d)
            if  Profiling:
                prof=prof2(X_used,y,i)
                p=prof3('continuous',prof,i,'',obs,mean_y=average_all)  
                p.to_csv(self.out,header=None,index=False,mode='a+',encoding='utf8')
        return self
    
        
    def transform(self,X_test):        
        d2=X_test.loc[:,self.use_cols].astype(np.float32)  
        
        #Consider case where  X contains ordinal_str columns
        if hasattr(self,'str_cols'):
            if  self.order_cats is not None:
                for j in self.order_cats:
                    d2.loc[:,j]=d2.loc[:,j].astype(CategoricalDtype(categories=self.order_cats[j])).cat.codes
                d2.replace(to_replace=-1,value=np.nan,inplace=True)
            else:
                d2.loc[:,self.str_cols]=d2.reindex(columns=self.str_cols).astype(np.float32)
                
        df=pd.DataFrame()
        for i in self.use_cols:
            if  self.fit[i]['new_var']  !='':
                #Attention!  d2  is a full dataframe  not a single variable  series
                var_dict=self.fit[i]
                d2.fillna({i:var_dict['var_miss']},inplace=True)   
                d2.loc[:,i]=np.clip(d2.loc[:,i],var_dict['var_lb'],var_dict['var_ub'])  
                if var_dict['new_var'][:3]  in  trans_prefix:
                    new_var_name=myprefix+var_dict['new_var']
                    d2.loc[:,i]=transform_cont(d2.loc[:,i],var_dict['new_var'][:3])  
                else:
                    new_var_name=myprefix+i
                if self.standardtransform:
                    d2.loc[:,i]=var_dict['scale'].transform(d2.loc[:,i].values.reshape(-1,1))  
                df[new_var_name]=d2.loc[:,i]   
            else:
                continue

        return  df
            
    
class  re_ordi(re_cont): 
    """Transformer for ordinal varibles,generate recoded rule for each variable and output Profiling file and correlation_with_y fil
            Args:
                ordi_variables:sequence,ordinal variables
                standardtransform:boolean,if standardtransform is True,default False
                out:str,path for output Profiling file
                
        Return:
            fit():instance,self
            transform():DataFrame,result dataframe
            
        Methods:
            see sklearn.base.TransformerMixin or  simply  sklearn.linear_model.LogisticRegression
    """
    def __init__(self,ordi_variables,out):
        self.vars=ordi_variables
        self.standardtransform=standardtransform_O
        self.out =out
        self.order_cats=order_cats
        
    def fit(self,X,y=None):
      
        # x :pandas.dataframe,y:array  type;out:  output path
        # add var_miss,var_lb,var_ub,new_variable,standardtransformation(mean,std)
        self.use_cols=X.loc[: ,self.vars].apply(c_ordi)[lambda i:i== True].index
        self.fit={}
        
        X_used=X.reindex(columns=self.use_cols) 
        #Consider one case where X_used DataFrame containing str columns
        self.str_cols= X_used.select_dtypes(include =['object']).columns
        if len(self.str_cols)>0:
            if self.order_cats is not None:
                for j in self.order_cats:
                    X_used.loc[:,j]=X_used.loc[:,j].astype(CategoricalDtype(categories=self.order_cats[j])).cat.codes
                X_used.replace(to_replace=-1,value=np.nan,inplace=True)
            
            else:
                X_used.loc[:,self.str_cols]=X_used.reindex(columns=self.str_cols).astype(np.float32)
            
        for i in self.use_cols:
            d=pnum(X_used,y,cont_var= i,impmethod=impmethodO,transf=transf_O,standardtransform=standardtransform_O,cap_floor=cap_floor_O)
            self.fit.update(d)
            if Profiling:
                if len(X_used.loc[:,i].unique())<=num_category:
                    prof=prof1(X_used,y,i)
                else:    
                    prof=prof2(X_used,y,i)
                p=prof3('ordinal',prof,i,'',obs,mean_y=average_all)  
                p.to_csv(self.out,header=None,index=False,mode='a+',encoding='utf8')
        return self
    


class  re_norm(BaseEstimator,TransformerMixin):
    """Transformer for nominal varibles,generate recoded rule for each variable and output Profiling file and correlation_with_y fil
            Args:
                norm_variables:sequence,nominal variables
                nom_method:{'binary_encoder','binary','index'}
                out:str,path for output Profiling file
                
        Return:
            fit():instance,self
            transform():DataFrame,result dataframe
            
        Methods:
            see sklearn.base.TransformerMixin or  simply  sklearn.linear_model.LogisticRegression
    """
    def  __init__(self,norm_variables,out):
        self.vars=norm_variables
        self.out=out
        self.nom_method=nom_method.lower() 
        
    def fit(self,X,y=None):   
        self.use_cols= X.loc[: ,self.vars].apply(c_norm)[lambda x: x== True].index
        self.fit={}
        
        X_used=X.loc[:,self.use_cols].astype(str)
        X_used.replace(to_replace='nan',value='missing',inplace=True)
        
        if self.nom_method !='binary_encoder':
            for i in self.use_cols:    
                tmp,d=pnom(X_used,y,i,self.nom_method)  
                self.fit.update(d)
                if Profiling:
                    prof=prof4(tmp,i) 
                    p=prof3('normial',prof,i,'',obs,average_all)  
                    p.to_csv(self.out,header=None,index=False,mode='a+',encoding='utf8')
        else: 
            self.fit=ce.BinaryEncoder(X_used,cols=self.use_cols,drop_invariant=True,return_df=True).fit(X_used,y)
            if Profiling:
                print ('Cannot output Profile  since nom_method  is binary_encoder')
        return self
    
    def transform(self,X_test):
        d2=X_test.loc[:,self.use_cols].astype(str)   
        d2.replace(to_replace='nan',value='missing',inplace=True)
        
        if self.nom_method=='binary':
            df=pd.get_dummies(d2.replace(to_replace=self.fit),prefix='',prefix_sep='')
        else:
            if self.nom_method=='binary_encoder':
                df=self.fit.transform(d2)
            else:
                df=d2.replace(to_replace=self.fit)
            df.rename(columns=lambda x:myprefix+x,inplace=True)
        return  df 


    
class  re_bina(BaseEstimator,TransformerMixin):
    """Transformer for binary varibles,output Profiling file and correlation_with_y fil
        Args:
            bina_variables:sequence,binary variables
            out:str,path for output Profiling file
            
    Return:
        fit():instance,self
        transform():DataFrame,result dataframe
        
    Methods:
        see sklearn.base.TransformerMixin or  simply  sklearn.linear_model.LogisticRegression
    """
    
    def __init__(self,bina_variables,out):
        self.vars=bina_variables
        self.out=out
        
    def fit(self,X,y):
        self.use_cols= X.loc[: ,self.vars].apply(c_bina)[lambda x: x== True].index
        X_used=X.reindex(columns=self.use_cols)
        
        if Profiling:
            if len(self.use_cols)==0:
                print('No Binary  columns selected')
            else:
                for i in self.use_cols:
                    prof=prof5(X_used,y,i)
                    p=prof3('binary',prof,i,'',obs,average_all)
                    p.to_csv(self.out,header=None,index=False,mode='a+',encoding='utf8')
        return self
        
                
    def transform(self,X_test):
        d2= X_test.loc[:,self.use_cols]
        return   pd.DataFrame(np.where(d2.applymap(lambda x: x in [1,'1','y','Y']),1,0),columns=[myprefix+i  for i in self.use_cols])


    
class  Recoded(BaseEstimator,TransformerMixin):
    """Synthetic estimator contains in  parallel four meta-transformer {'re_cont','re_ordi','re_norm','re_bina'},recode the original-data 
        according to different  dtype
        Args:
        ----------
        
        type_cols:dict or  mapping type,variable sets {'cont_cols':list,'ordi_cols':list,'bina_cols':list,'norm_cols':list},
                  keys name:{'cont_cols','ordi_cols','norm_cols','bina_cols'},not necessary to  include all keys name 
        profiling: boolean,if output Profiling file
        write_corr:boolean,if output correlation file
        write_corr_method: str,  {‘pearson’, ‘kendall’, ‘spearman’}
        
        
        y_dist:str,{'CLASSES,REGRESSION'};distribution of  y
        miss_cnt:float 0-1,default 0.99 ; maximum missing rate allowed  

        out: str;path of output (Profling and correlation)
        transformer_weights : ist of (string, transformer) tuples
                                List of transformer objects to be applied to the data. The first half of each tuple is the name of the transformer.
        n_jobs: int, default 1 ;
                Number of jobs to run in parallel (default 1).
        random_state:default np.random.RandomState(0);random_seed
        
        prefix:str ,default 'f_';variable recoding prefix        
         
         
        
        #Continuous and Ordinal variable types
        P: float,default 0.05;Pvalue threshold to include variable
        equal_dist:str {'No','YES','yes'},default 'yes'; Use equal distance for dividing variables into groups for profiling (Y/N)
        num_category:int,default 5;Maximum number of categories for profiling variables
        min_size:int,default 10; Minimum missing group size for Equal Response imputation
     
            #Continuous variable types
            
            p_lo: float 0-1,default 0.01; Lower percentile for checking constant value
            p_hi: float 0-1,default 0.99; Upper percentile for checking constant value
            impmethodC: str,{'mean','std'，'median', 'iqr','mad'，'range'，'midrange'，'sum','euclen','ustd','maxabs'},default 'mean'
                        What method to use for missing imputation for continuous variables
            transf_C: boolean,default True;Include transformed variables in evaluation (Y/N)
            standardtransform_C: boolean,default True; Standardization options: see sklearn.preprocessing.StandardScaler
            cap_floor_C: boolean,default True;Do you want to do capping / flooring to handle outliers? (Y/N)
            
            #Ordinal variable types
            
            impmethodO: str,{'mean','std'，'median', 'iqr','mad'，'range'，'midrange'，'sum','euclen','ustd','maxabs'},default 'median'
                        What method to use for missing imputation for ordinal variables
            transf_O: boolean,default False;Include transformed variables in evaluation (Y/N)
            standardtransform_O: boolean,default False; Standardization options: see sklearn.preprocessing.StandardScaler
            cap_floor_O: boolean,default False;Do you want to do capping / flooring to handle outliers? (Y/N)
            order_cats：nestd dict ,Categorical variable;e.g:{col_name:[ordered catgories]}
                        this ensures str value in Categorical variable converted to numeric values in a  customized order

        #Binary and Nominal variable types;
        concrate: float 0-1,default 0.5; Maximum amount of  count that can be in a single value of an independent variable
            
            #Nominal variable types
                valcount: int,default 10;maximum number of unique values allowed. Set to 0 to allow any number
                minbinn: int,default 100;Minimum count in a bin to be usable. 
                bonfer:str {'No','YES','yes'},default 'yes';Do Bonferoni adjustment for talpha? (Y/N)
                talpha: float,default 0.05;T Test significance level for collapse of bins
                nom_method: str {'binary','binary_encoder','index'};Recoding method for nominal variables
    
    Attributes:
        combined_features:MyFeatureUnion instance fitted;see MyFeatureUnion class
        selected_columns:list of str;list of columns choosed after fit method
        
        
    
   
    """
    def __init__(self,type_cols=None,profiling=True,Y_dist='CLASSES',Order_cats=None,P_lo=0.01,P_hi=0.99,Concrate=0.5,Miss_cnt=0.99
                 ,p=0.05,Equal_dist='yes',Num_category=5,
                   ImpmethodC='mean' ,Transf_C=True,Standardtransform_C=1,Cap_floor_C=True,Min_size=10,
                   ImpmethodO='median',Transf_O=False,Standardtransform_O=False,Cap_floor_O=False,
                   Valcount=10, Minbinn=10,Bonfer='yes',Talpha=0.05 ,Nom_method='index',
                   out=None,transformer_weights=None,n_jobs =1,Prefix='f_',
                   write_corr=True,write_corr_method='spearman'):
                
        global Profiling,y_dist,p_lo,p_hi,concrate,miss_cnt,P,equal_dist,num_category,\
           impmethodC ,transf_C,standardtransform_C,cap_floor_C,min_size,\
           impmethodO,transf_O,standardtransform_O,cap_floor_O,order_cats,\
           valcount, minbinn,bonfer,talpha,nom_method,myprefix
        
        self.type_cols= type_cols
        self.transformer_weights=transformer_weights
        self.n_jobs=n_jobs
        self.out=out
        self.write_corr= write_corr
        self.write_corr_method= write_corr_method

                 
        Profiling=profiling
        y_dist=Y_dist
        order_cats=Order_cats
        p_lo=P_lo
        p_hi=P_hi
        concrate=Concrate  
        miss_cnt=Miss_cnt
        P=p
        
        equal_dist=Equal_dist 
        num_category=Num_category
        
        impmethodC=ImpmethodC  
        transf_C=Transf_C
        standardtransform_C=Standardtransform_C
        cap_floor_C=Cap_floor_C
        min_size=Min_size

        impmethodO=ImpmethodO
        transf_O=Transf_O
        standardtransform_O=Standardtransform_O
        cap_floor_O=Cap_floor_O
    
        valcount=Valcount
        minbinn=Minbinn
        bonfer=Bonfer
        talpha=Talpha       
        nom_method=Nom_method   
      
        myprefix=Prefix                   
                  
        
        
    def fit(self,X,Y):


        global profile_cols,trans_prefix,min_y,max_y,average_all,myprefix,obs,y
        assert len(X)==len(Y),ImportError('X must be the same length with y')
        
        profile_cols=('type','variable','label','category','count','percent','Average_DV','index','star')
        trans_prefix=('SQ_','SR_','LN_')
        min_y=min(Y)
        max_y=max(Y)
        average_all=np.mean(Y)
        obs=len(Y)
        y=Y
        self.y=Y
        columns={'cont_cols','ordi_cols','norm_cols','bina_cols'}
        
        assert len(columns.intersection(self.type_cols.keys()))!=0,ValueError('type_cols keys not standard')
        
        try:
            re_c=re_cont(self.type_cols['cont_cols'],os.path.join(self.out,'ce2_cont_test.csv'))
        except KeyError:
            re_c=None
        try:
            re_o=re_ordi(self.type_cols['ordi_cols'],os.path.join(self.out,'ce2_ordi_test.csv'))        
        except KeyError:
            re_o=None
        try:
            re_n=re_norm(self.type_cols['norm_cols'],os.path.join(self.out,'ce2_norm_test.csv'))
        except KeyError:
            re_n=None
        try:
            re_b=re_bina(self.type_cols['bina_cols'],os.path.join(self.out,'ce2_bina_test.csv'))
        except KeyError:
            re_b=None
     
        
        self.combined_features=MyFeatureUnion(transformer_list=[('re_cont',re_c),
                                 ('re_ordi',re_o),
                                 ('re_norm',re_n),
                                 ('re_bina',re_b)],n_jobs=self.n_jobs)
        self.combined_features.fit(X,y)
        return self
    
    def transform(self,X):
        df=self.combined_features.transform(X)
        if not hasattr(self,'selected_columns'):
            self.selected_columns=df.columns.tolist()
        if self.write_corr and not os.path.isfile(os.path.join(self.out,'corr.xlsx')):
            obj_cat_cols=X.select_dtypes(include=['object','category']).columns
            X_dummies=pd.get_dummies(X,columns=obj_cat_cols)
            writecorr(X_dummies,df,self.y,out=os.path.join(self.out,'corr.xlsx'),method=self.write_corr_method)
        return  df


class FSBMD(BaseEstimator,TransformerMixin):
    """feature selection based on model and decomposition estimator
            Args:
                n_components:int; number of component for decomposition estimator,see skleanr.discriminant_analysis.lda if y_dist=='classes'  or sklearn.decomposition.PCA  if y_dist='regression'
                y_dist:{'classes','regression'}; distribution of target data 
                scoring:string, callable or None, optional, default: None
                        A string (see model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y).
                n_estimators: int ;number  of  estimator in nested tree_based model
                rfe:class;see class FSBM.te args
                step:see MyRFECV.step  param
                cv:see MyRFECV.cv param
                transformer_weights:see sklearn.pipeline.Feature_Union args
        Attributes:
            selected_columns:list of str;list of columns choosed after fit method
        
        Method:
            get_dtransformer:return decompostion estimator,detail  see skleanr.discriminant_analysis.lda if y_dist=='classes'  or sklearn.decomposition.PCA  if y_dist='regression'
    
    """
    def  __init__(self,y_dist='Classes',n_components=None,n_estimators=5,rfe=RFECV,step=1,cv=3,scoring='auto',transformer_weights=None):
        self.n_components=n_components
        self.y_dist=y_dist
        self.scoring=scoring
        self.n_estimators=n_estimators
        self.rfe=rfe 
        self.step=step
        self.cv=cv
        self.transformer_weights=transformer_weights
        
    
                
    def fit(self,X,y):
        
        if self.y_dist.lower()=='classes':
            self.decomposition_estimator=LinearDiscriminantAnalysis(solver='svd',n_components=self.n_components)
            rlr=RandomizedLogisticRegression(n_resampling=20)
            rfc=XGBClassifier(n_estimators=self.n_estimators)
            if self.scoring=='auto':
                scoring=make_scorer(auc_mc)

                
        else:
            self.decomposition_estimator=PCA(n_components=0.99 if self.n_components is None else self.n_components,svd_solver ='full')
            rlr=RandomizedLasso(n_resampling=20)
            rfc=XGBRegressor(n_estimators=self.n_estimators)
            if scoring=='auto':
                scoring=make_scorer(accuracy_score)
        
        if not locals().get('scoring'):
            scoring=make_scorer(self.scoring)
            
            
        rfe=self.rfe(rfc,step=self.step,cv=self.cv,scoring=scoring)   #Selectfrommodel,myrfe
        final_feature_union=MyFeatureUnion(transformer_list=[('decomposition',make_pipeline(StandardScaler(),PCA(),self.decomposition_estimator)),
          ('fsbm',FSBM(rlr,rfe))],transformer_weights=self.transformer_weights,n_jobs=1) 
    
        
        self.model_fit=final_feature_union.fit(X,y)
        return self
    
    def transform(self,X):
        df=self.model_fit.transform(X)
        if not hasattr(self,'selected_columns'):
            self.selected_columns= df.columns.tolist()
        return df
    
    @property
    def get_dtransformer(self):
        return self.decomposition_estimator
    
def set_params(estimator,dtype,var=None,attr=None,value=None):
    """Inspect all var name corresponding to special dtype,all attributes value corresponding to one given variable;
    or update value of one given attribute  
            Args:
            estimator: fitted instance,CE  instance fitted
            dtype:str ,{'re_cont','re_bina','re_norm','re_ordi'} ;variable dtype 
            attr:re_cont,re_ordi:{'new_var','scale','var_lb','var_miss','var_ub'}
                 re_norm:binary_encoder raise  AttributeError()
                         Index,binary:{'categorie':value}  ,{'categorie':new_var} 
            value: see  set_params.attr descriptions
    """    
    
    combined_f=estimator.named_steps['recoded'].combined_features
    
    if var is None:
        #print all columns names from the same dtype
        print('{0} contains {1} columns'.format(dtype,list(combined_f.get_params(deep=True)[dtype].fit.keys())))
    elif attr is None:
        #print all attr of var
        print('{0} columns contains {1} attrs'.format(var,combined_f.get_params(deep=True)[dtype].fit[var]))
    elif value is None:
        print("{0} columns 's  attr {1} :{2}".format(var, attr,combined_f.get_params(deep=True)[dtype].fit[var][attr]))
    else:
        combined_f.get_params(deep=True)[dtype].fit[var][attr]=value
        
        
    return 
       
    

        







