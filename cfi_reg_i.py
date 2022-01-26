# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 09:52:33 2021

@author: nicola picchiotti
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pl
import fairness
import matplotlib
font={'size':20}
matplotlib.rc('font', **font)
pd.set_option('display.max_columns', None)

file_path = 'adult.data'
col = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
 'marital-status', 'occupation', 'relationship', 'race',
'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
 'country', 'salary']
df = pd.read_csv(file_path,",")
df.columns = col
salary_map={' <=50K':0,' >50K':1}
df['salary']=df['salary'].map(salary_map).astype(int)
df['sex'] = df['sex'].map({' Male':1,' Female':0}).astype(int)
df['country'] = df['country'].replace(' ?',np.nan)
df['workclass'] = df['workclass'].replace(' ?',np.nan)
df['occupation'] = df['occupation'].replace(' ?',np.nan)
df.dropna(how='any',inplace=True)
df.loc[df['country'] != ' United-States', 'country'] = 'Non-US'
df.loc[df['country'] == ' United-States', 'country'] = 'US'
df['country'] = df['country'].map({'US':1,'Non-US':0}).astype(int)
df['marital-status'] = df['marital-status'].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')
df['marital-status'] = df['marital-status'].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')
df['marital-status'] = df['marital-status'].map({'Couple':1,'Single':0})
rel_map = {' Unmarried':0,' Wife':1,' Husband':2,' Not-in-family':3,' Own-child':4,' Other-relative':5}
df['relationship'] = df['relationship'].map(rel_map)
race_map={' White':1,' Amer-Indian-Eskimo':0,' Asian-Pac-Islander':0,' Black':0,' Other':0}
df['race'] = df['race'].map(race_map)
def f(x):
    if x['workclass'] == ' Federal-gov' or x['workclass']== ' Local-gov' or x['workclass']==' State-gov': return 'govt'
    elif x['workclass'] == ' Private':return 'private'
    elif x['workclass'] == ' Self-emp-inc' or x['workclass'] == ' Self-emp-not-inc': return 'self_employed'
    else: return 'without_pay'
df['employment_type']=df.apply(f, axis=1)
employment_map = {'govt':0,'private':1,'self_employed':2,'without_pay':3}
df['employment_type'] = df['employment_type'].map(employment_map)
df.drop(labels=['workclass','education','occupation'],axis=1,inplace=True)
df.loc[(df['capital-gain'] > 0),'capital-gain'] = 1
df.loc[(df['capital-gain'] == 0 ,'capital-gain')]= 0
df.loc[(df['capital-loss'] > 0),'capital-loss'] = 1
df.loc[(df['capital-loss'] == 0 ,'capital-loss')]= 0

np.random.seed(999)

df = df.reset_index()
def balanced_subsample(y):
    subsample = []
    n_smp = y.value_counts().min()
    for label in y.value_counts().index:
        samples = y[y == label].index.values
        index_range = range(samples.shape[0])
        indexes = np.random.choice(index_range, size=n_smp, replace=False)
        subsample += samples[indexes].tolist()
    return subsample
df = df.iloc[balanced_subsample(df['salary'])]
df = df.reset_index()
df = df.iloc[balanced_subsample(df['race'])]
df = df.sample(frac=1)

X = df
y = df['salary']

protected_attribute = 'race'
label = 'salary'

unprivileged_groups=[{protected_attribute: [0.]}]
privileged_groups=[{protected_attribute: [1.]}]
df_X = X
df_X = df_X[['race','marital-status','sex', 'education-num', 
             'age', 'fnlwgt', 'relationship',
              'capital-gain', 'capital-loss', 'hours-per-week',
             'country', 'employment_type', 'salary']]

I_constraint = np.zeros(df_X.shape[1]-1)
rho = np.abs(df_X.corr().values)
reg_correlated = rho[0, :-1]


df_train, df_test = train_test_split(df_X, train_size=0.5, shuffle=True,
                                         stratify=df_X[[label, protected_attribute]], random_state=42)#42
columns_num = ['age', 'fnlwgt', 'education-num',
               'relationship', 'hours-per-week', 'employment_type']
scaler = MinMaxScaler().fit(df_train[columns_num])
df_train[columns_num] = scaler.transform(df_train[columns_num])
df_test[columns_num] = scaler.transform(df_test[columns_num])

y_train = df_train[label]
X_train = df_train.drop(label, axis=1)
y_test = df_test[label]
X_test = df_test.drop(label, axis=1)

df_r = pd.DataFrame()
reg_list = np.linspace(0, 0.5, 10)
for i_reg in reg_list:
    print(i_reg)

    from constraints_sklearn_nn import Nn_model
    result = []
    model = Nn_model(HIDDEN_UNITS=4, n_fea=12)
    model.fit(X_train, y_train, IMP_REG_FACTOR=i_reg*reg_correlated,
              I_constraint=I_constraint,
              LEARNING_RATE=0.1, EPOCHS=10)

    res = fairness.compute_metrics(model, X_test, y_test, df_test, 
                                   unprivileged_groups, privileged_groups,
                                   protected_attribute, label)

    res['name'] = 'nn'
    result.append(res)
    
    pl.figure('importances')
    pl.plot(model.importance(X_test, y_test), label=i_reg, linewidth=3)
    pl.grid()
    pl.legend()

    df_result = pd.DataFrame(result)
    df_result = df_result.set_index('name').reset_index()

    df_r = pd.concat((df_r, df_result))



pl.figure('Comparison')

pl.subplot(4, 1, 1)
x, y = reg_list, df_r['disp_impact'].values
pl.plot(x, y,'ko-')
pl.ylabel('DI')
pl.tick_params(labelbottom=False)    

pl.subplot(4, 1, 2)
x, y = reg_list, df_r['avg_odds'].values
pl.plot(x, y,'ko-')
pl.ylabel('EO')
pl.tick_params(labelbottom=False) 

x, y = reg_list, df_r['counterfactual'].values
pl.subplot(4, 1, 3)
pl.plot(x, y,'ko-')
pl.tick_params(labelbottom=False) 
pl.ylabel('CF')

x, y = reg_list, df_r['roc_auc'].values
pl.subplot(4, 1, 4)
pl.plot(x, y,'ko-')
pl.xlabel('regularization strength')
pl.ylabel('ROC-AUC')
pl.ylim(0.75, 0.83)

