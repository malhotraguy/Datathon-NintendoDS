#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:21:31 2019

@author: RahulMalhotra
"""
#importing libaries
import pandas as pd
import numpy as np

#importing files
df=pd.read_excel('/home/rahul/PycharmProjects/Datathon/DIP_Reduced_data.xlsx')
ref_df=pd.read_excel('/home/rahul/PycharmProjects/Datathon/PRIZM5_2018_Reference.XLSX')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PREPROCESSING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
summary_ref=ref_df.describe() #Households with highest average income have the highest % of children

#checking for NA values
df.isna().any().any()
ref_df.isna().any().any()

df=df[df['PRIZM5DA'] != 'UN'] #removing junk observations

ref_df=ref_df.filter(['SESI','Name','PRIZM5 Descriptor','Cultural Diversity Index','Age of Children '])
ref_df.columns=['PRIZM5DA', 'Name', 'PRIZM5_Descriptor', 'Cultural_Diversity_Index', 'Age_of_Children']


#Values of children's ages are hard to classify; observe impact on having a young child (<15)
no_young_child=['Mixed', '20+', '15+', '<25', '<20']
ref_df['Age_of_Children_Young']=0
for i in range(len(ref_df['Age_of_Children_Young'])):
    for j in range(len(no_young_child)):
        if no_young_child[j] in ref_df['Age_of_Children'][i]:
            ref_df['Age_of_Children_Young'][i]=1
ref_df['Age_of_Children_Young']=np.where(ref_df['Age_of_Children_Young']==1,0,1) #reversing values

ref_df=pd.get_dummies(columns=['Cultural_Diversity_Index'],data=ref_df)
ref_df=ref_df.drop(['Cultural_Diversity_Index_Low','Age_of_Children'],axis=1) #avoiding dummy trap & avoiding duplicate types of information

df['PRIZM5DA']=pd.to_numeric(df['PRIZM5DA']) #converting column PRIZM5DA to numeric datatype
df=ref_df.merge(right=df,how='right',on='PRIZM5DA')

df=pd.get_dummies(columns=['DensityClusterCode5_lbl','DensityClusterCode5','DensityClusterCode15'],data=df)

df_backup=df.copy() #creating backup of df in case

df=df.drop(['SG','LS','Name','PRIZM5_Descriptor','PRCDDA','DensityClusterCode5_lbl_Exurban','DensityClusterCode5_1','DensityClusterCode15_1'],axis=1)

summary_df=df.describe()

'''---------------------------------------------Deliverable #1---------------------------------------------'''

DA_grouped=df.groupby('PRIZM5DA',as_index=False).mean()
DA_grouped=DA_grouped.filter(['PRIZM5DA','DEPVAR7','CNBBAS19P'],axis=1)

i_selected=[]
for lower in [[np.floor(DA_grouped['DEPVAR7'].max()/4 * (i)) for i in range(4)]]:
    for upper in [[np.floor(DA_grouped['DEPVAR7'].max()/4 * (i+1)) for i in range(4)]]:
        for x in range(len(DA_grouped['DEPVAR7'])):
            for i in range(4):
                if i==3:
                    if lower[i]<=np.floor(DA_grouped['DEPVAR7'][x]): #making it inclusive for last iteration (max value)
                        i_selected.append(i+1)
                else:
                    if lower[i]<=np.floor(DA_grouped['DEPVAR7'][x])<upper[i]:
                        i_selected.append(i+1)
DA_grouped['class']=i_selected

#deliverable 1
freq_dist=pd.DataFrame({'Class':[1,2,3,4],'Class_Bound-Exp_Usage':[str(np.floor(DA_grouped['DEPVAR7'].max()/4 * (i)))+' - '+str(np.floor(DA_grouped['DEPVAR7'].max()/4 * (i+1))) for i in range(4)],
                        'Number of DA':np.unique(i_selected, return_counts=True)[1],
                        'Avg Number of Adults (19+)': DA_grouped['CNBBAS19P'].groupby(DA_grouped['class']).mean()})

    
'''------------------------------------------------------------------------------------------------------------'''

df=pd.get_dummies(columns=['PRIZM5DA'],data=df).drop(['PRIZM5DA_1'],axis=1)
df_new=df.copy()

'''---------------------------------------------Deliverable #2---------------------------------------------'''

#removing multicollinearity

multico_var=[]
corr=df_new.corr()
for i in range(corr.shape[1]):
    for j in range(corr.shape[1]):
        if i!=j:
            if np.abs(corr.iloc[i,j])>=.65:
                multico_var.append(corr.columns[i])
multico_var=list(np.unique(multico_var))
multico_var.remove('DEPVAR7')
multico_df=df_new.filter(multico_var)

multico_var_index=multico_var.copy()
for i in multico_var_index:
    if (multico_df.loc[:,i] >=.65).any():
        multico_var.remove(i)

df_new=df_new.drop(list(set(multico_var_index)-set(multico_var)),axis=1)


#Splitting dependent & independent variables

X=df_new.drop('DEPVAR7',axis=1)
y=df_new.iloc[:,3:4]

from scipy.stats import pearsonr
coeffmat = np.zeros((X.shape[1], y.shape[1]))
pvalmat = np.zeros((X.shape[1], y.shape[1]))

for i in range(X.shape[1]):
    for j in range(y.shape[1]):
        test=pearsonr(X[X.columns[i]],y[y.columns[j]])
        coeffmat[i,j] = test[0]
        pvalmat[i,j] = test[1]

corr=pd.DataFrame({'col_name':X.columns,'coef':np.ravel(coeffmat),'p-value':np.ravel(pvalmat)})
corr=corr[corr['p-value']<0.01].sort_values(['coef'],ascending=False)

#deliverable 2
top_10_var=corr[:10]
top_20_var=corr[:20]
list(top_20_var.iloc[:,0])

#df_new.to_excel(pd.ExcelWriter('clean_data.xlsx'))
'''------------------------------------------------------------------------------------------------------------'''

from sklearn.model_selection import train_test_split
development_file,validation_file=train_test_split(df,test_size=.5,random_state=0)

'''-------------------------------------------Stepwise Regression---------------------------------------------------'''

dev_df=development_file.copy()
val_df=validation_file.copy()

X=dev_df.filter(list(top_20_var.iloc[:,0]))
y=dev_df.loc[:,'DEPVAR7']

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,r2_score

cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.01):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features = cols
print(selected_features)

X=X.filter(selected_features)
#X=sm.add_constant(X)
linear_reg=sm.OLS(y,X).fit()
linear_reg.summary()

X_val=val_df.filter(selected_features)
y_val=val_df.loc[:,'DEPVAR7']
y_pred=linear_reg.predict(X_val)
mean_squared_error(y_pred,y_val)
r2_score(y_val,y_pred)


