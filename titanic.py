# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:21:17 2021

@author: nicol
"""

import pandas as pd
import numpy as np
import seaborn as sns
import copy
import matplotlib as mpl
import matplotlib.pyplot as pl
import math
import random
import tensorflow as tf
from sklearn.utils import shuffle
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from keras.activations import relu
from keras.activations import sigmoid
sns.set(font_scale=1.1)
siz = 12
mpl.rcParams['xtick.labelsize'] = siz
mpl.rcParams['axes.labelsize'] = siz
mpl.rcParams['font.size'] = siz
mpl.rcParams['axes.labelsize'] = siz
pd.get_option('display.max_columns')

#################################################################
#                   pre processing phase
#################################################################

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]
    dataset['Age'] = dataset['Age'].astype(int)
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

feature_names = X_train.columns
X_train = X_train.values
y_train = Y_train
X_train, _, y_train = shuffle(X_train, X_train, y_train, random_state=123)


#################################################################
#                   cluster interpretation
#################################################################

random.seed(0)
tf.random.set_seed(0)

p = 2 # number of relu layer
n1 = 3 # neurons in the first layer
n2 = 2 # neurons in the second layer
n_neurons = n1 + n2

def nn_model(best_param_nn_l1=0, ratio_l1_l2=0, best_param_activity=0):
    mod = Sequential()
    mod.add(Dense(n1, use_bias=True, input_dim=X_train.shape[1], 
                  activation='relu', kernel_regularizer=l2(best_param_nn_l1),
                  activity_regularizer=l1(best_param_activity)))
    mod.add(Dense(n2, use_bias=True, activation='relu'))
    mod.add(Dense(1, use_bias=True, activation='sigmoid'))
    opt=Adam()
    mod.compile(loss='binary_crossentropy', optimizer=opt,
                metrics=['accuracy'])
    return mod
    
keras_model = nn_model(best_param_nn_l1=0.1)

history = keras_model.fit(X_train, y_train, epochs=100, validation_split=0.1)
print('performance on train set: ', keras_model.evaluate(X_train, y_train))

pl.figure('loss as a function of epochs')
pl.plot(history.history['loss'], label='training')
pl.plot(history.history['val_loss'], label='validation')
pl.legend()

pl.figure('accuracy as a function of epochs')
pl.plot(history.history['accuracy'], label='training')
pl.plot(history.history['val_accuracy'], label='validation')
pl.legend()

df_fi = pd.DataFrame(history.model.weights[0].numpy()[:, 0], index=feature_names)

# collect weights and bias
W = []
for i_relu_layer in np.arange(p+2)[::2]:
    W.append(keras_model.weights[i_relu_layer].numpy())
W.append(keras_model.weights[p+2].numpy())

B = []
for i_relu_layer in np.arange(1,p+3)[::2]:
    B.append(np.expand_dims(keras_model.weights[i_relu_layer].numpy(), 0))
B.append(keras_model.weights[(p*2)+1].numpy())

# compute activations/z
A = []
Z = []
for k in range(p):
    if k == 0:
        A.append(np.dot(X_train, W[0]) + B[0])
        Z.append(relu(A[0]).numpy())
    else:
        A.append(np.dot(Z[k-1], W[k]) + B[k])
        Z.append(relu(A[k]).numpy())
A.append(np.dot(Z[p-1], W[p]) + B[p])
Z.append(sigmoid(A[p]))

# check activations/z
pred = keras_model.predict(X_train)
assert np.round(Z[-1].numpy().sum(), 0) == np.round(pred.sum(), 0)

# compute effective weights/bias
W_eff = np.zeros(X_train.shape)
for en, x in enumerate(X_train):
    w_temp = W[0]
    for k in range(p):
        w_temp = np.dot(w_temp * (A[k][en]>0), W[k+1])
    W_eff[en, :] = w_temp.T[0]
    
B_eff = np.zeros(X_train.shape[0])
for en, x in enumerate(X_train):
    for k in range(0, p):
        b_temp = B[k]
        for h in range(k, p):
            b_temp = np.dot(b_temp * (A[h][en]>0), W[h+1])
        B_eff[en] = B_eff[en] + copy.deepcopy(b_temp)[0]
B_eff = B_eff + B[p][0]

# check effective weights
zz = np.zeros(X_train.shape[0])
for g in range(X_train.shape[0]):
    zz[g] = np.dot(X_train[g], W_eff[g]) + B_eff[g]

def sigmoid_(x):
  return 1 / (1 + math.exp(-x))
pred_effective = [sigmoid_(zz_) for zz_ in zz]
assert np.round(Z[-1].numpy().sum(), 0) == np.round(np.sum(pred_effective), 0)

# create clusters based on Z
mat = Z[0]
for z in range(1, p):
    mat = np.hstack((mat, Z[z]))
mat = mat>0
clu = mat.dot(1 << np.arange(mat.shape[-1]))

df_res = pd.DataFrame()
df_metaresults = pd.DataFrame()
for en, kk in enumerate(np.unique(clu)):
    explaination_ = W_eff[clu==kk, :]
    pred_cluster = 1*(keras_model.predict(X_train)[clu==kk]>0.5)
    positive = np.round(np.sum(pred_cluster)/pred_cluster.shape[0], 2)
    df_res['Feature'] = [nam for nam in feature_names]
    df_res['Feature Importance'] = np.mean(explaination_, 0)
    df_res['Abs score'] = np.abs(df_res['Feature Importance'])
    df_plot = df_res.sort_values(['Abs score'], ascending=False)
    seq = "{0:b}".format(kk).zfill(n_neurons)[::-1]
    df_metaresults.loc[seq, 'Support'] = pred_cluster.shape[0]/(X_train.shape[0])#np.round(pred_cluster.shape[0]/(n_samples*train_size),2)
    df_metaresults.loc[seq, 'Positive'] = positive
    df_metaresults.loc[seq, 'Avg instance'] = str(list( np.round((X_train)[clu==kk].mean(0), 2) )) #np.round(
    df_metaresults.loc[seq, feature_names] = df_res['Feature Importance'].values
    df_plot.iloc[:100].plot.bar(fontsize=12, x='Feature', y='Feature Importance', rot=90, legend=False)
    pl.ylabel('Feature Importance')
    pl.xlabel('')

print(df_metaresults.round(2))

Y_pred = 1*(keras_model.predict(X_test)>0.5)[:, 0]

