# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:07:04 2020

@author: nicola picchiotti
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset
from constraints_sklearn_nn import Nn_model
import matplotlib.pyplot as pl
import fairness
pd.set_option('display.max_columns', None)

df_credit= pd.read_excel('german_credit_data.xlsx')
df_credit_sex = pd.get_dummies(df_credit['Sex'])
df_credit['Gender'] = df_credit_sex['male']
df_X = df_credit[['Gender', 'Age', 'Amount', 'Duration', 'Job', 'Risk']]
df_X['Job'] = 1*(df_X['Job'].isin([2, 3]))
df_X['Risk'] = 1*(df_X['Risk'].isin(['good']))

protected_attribute = 'Gender'
label = 'Risk'
dataset_name = 'german'
columns_num = ['Age', 'Amount', 'Duration']
unprivileged_groups=[{'Gender': [0.0]}]
privileged_groups=[{'Gender': [1.0]}]
I_constraint = np.ones(df_X.shape[1])
I_constraint[0] = 0
I_constraint = I_constraint[:-1]

I_constraint = np.zeros(df_X.shape[1]-1)
rho = np.abs(df_X.corr().values)
reg_correlated = rho[0, :-1]

def standard_scaler(df_train, df_test, columns_num):
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    scaler = StandardScaler().fit(df_train[columns_num])
    df_train_norm = scaler.transform(df_train[columns_num])
    df_train_norm = pd.DataFrame(df_train_norm, columns=columns_num)
    df_train_norm = df_train_norm.join(df_train.drop(columns_num, axis=1))

    df_test_norm = scaler.transform(df_test[columns_num])
    df_test_norm = pd.DataFrame(df_test_norm, columns=columns_num)
    df_test_norm = df_test_norm.join(df_test.drop(columns_num, axis=1))
    return df_train_norm, df_test_norm

def create_df_aif(df_train, df_test, label, protected_attribute, metadata):
    df_train_aif = BinaryLabelDataset(df = df_train, label_names=[label], 
                                      protected_attribute_names = [protected_attribute], 
                                      instance_weights_name=None, unprivileged_protected_attributes=[], 
                                      privileged_protected_attributes=[], metadata=metadata)
    
    df_test_aif = BinaryLabelDataset(df = df_test, label_names=[label], 
                                     protected_attribute_names = [protected_attribute], 
                                     instance_weights_name=None, unprivileged_protected_attributes=[], 
                                     privileged_protected_attributes=[], metadata=metadata)    
    return df_train_aif, df_test_aif

cols = df_X.columns
df_train, df_test = train_test_split(df_X, train_size=0.8, shuffle=True, 
                                     stratify=df_X[[label, protected_attribute]], random_state=42)
df_train, df_test = standard_scaler(df_train, df_test, columns_num)
df_train = df_train[cols]
df_test = df_test[cols]
df_train = df_train.set_index(protected_attribute).reset_index()
df_test = df_test.set_index(protected_attribute).reset_index()
X_train, y_train = df_train.drop(label, axis=1), df_train[label]
X_test, y_test = df_test.drop(label, axis=1), df_test[label]
df_train_aif, df_test_aif = create_df_aif(df_train, df_test, label, 
                                      protected_attribute, metadata=None)

result = []
model_name = 'nn'
model =  Nn_model(HIDDEN_UNITS=16, n_fea=5)

print('Model with bias')
model = Nn_model(HIDDEN_UNITS=16, n_fea=5)
method_name = 'orig'
model.fit(X_train, y_train, LEARNING_RATE=0.01, EPOCHS=10)
res = fairness.compute_metrics(model, X_test, y_test, df_test, unprivileged_groups, privileged_groups, protected_attribute, label)
name = '_'.join([dataset_name, model_name, method_name])
res['name'] = name
result.append(res)
pl.figure('importances')
pl.plot(model.importance(X_test, y_test), color='k', label='Without constraint', linewidth=3)
pl.xticks(np.arange(5), [tt[:10] for tt in X_test.columns], rotation=90)
pl.legend()

print('Model with CTFI')
model = Nn_model(HIDDEN_UNITS=16, n_fea=5)
method_name = 'cfi_02'
model.fit(X_train, y_train, IMP_REG_FACTOR=np.array([0.05, 0, 0, 0, 0]), I_constraint=I_constraint, LEARNING_RATE=0.01, EPOCHS=10)#0.4
res = fairness.compute_metrics(model, X_test, y_test, df_test, unprivileged_groups, privileged_groups, protected_attribute, label)
name = '_'.join([dataset_name, model_name, method_name])
res['name'] = name
result.append(res)
pl.figure('importances')
pl.plot(model.importance(X_test, y_test), color='g', label='CTFI gender', linewidth=3)
pl.legend()

print('Model with CTFI correlated')
model = Nn_model(HIDDEN_UNITS=16, n_fea=5)
method_name = 'cfi_03'
model.fit(X_train, y_train, IMP_REG_FACTOR=0.05*reg_correlated, I_constraint=I_constraint,
          LEARNING_RATE=0.01, EPOCHS=10)#0.4
res = fairness.compute_metrics(model, X_test, y_test, df_test, unprivileged_groups, privileged_groups, protected_attribute, label)
name = '_'.join([dataset_name, model_name, method_name])
res['name'] = name
result.append(res)
pl.figure('importances')
pl.plot(model.importance(X_test, y_test), color='r', label='CTFI gender and correlated', linewidth=3)
pl.legend()        

df_result = pd.DataFrame(result)
df_result = df_result.set_index('name').reset_index()

