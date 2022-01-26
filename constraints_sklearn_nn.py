# -*- coding: utf-8 -*-
"""
Created on Sun May 10 20:25:17 2020

@author: HO18971
"""

from torch.utils.data import Dataset, DataLoader
import torch as T
import torch.nn as nn
import numpy as np
import torch.optim as optim


T.manual_seed(0)

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

class binaryClassification(nn.Module):
    def __init__(self, HIDDEN_UNITS=None, n_fea=None):
        super(binaryClassification, self).__init__()
        self.layer_1 = nn.Linear(n_fea, HIDDEN_UNITS)
        self.layer_out = nn.Linear(HIDDEN_UNITS, 1)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = T.relu(x)
        x = self.layer_out(x)
        return x
        
    def predict(self,x):
        pred = self.forward(x)
        return pred

class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
class testData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
        
class Nn_model(object):

    def __init__(self, HIDDEN_UNITS=None, n_fea=None):
        self.model = binaryClassification(HIDDEN_UNITS=HIDDEN_UNITS,
                                          n_fea=n_fea)
    
    def fit(self, X_train, y_train, sample_weight=None,
            IMP_REG_FACTOR=None,
            I_constraint=None,
            LEARNING_RATE = None,
            EPOCHS = None):
        if sample_weight is not None:
            sample_weight = np.expand_dims(sample_weight/np.sum(sample_weight), 1)
            sample_weight = T.from_numpy(sample_weight)
            sample_weight = sample_weight.type(T.FloatTensor)
            
        if IMP_REG_FACTOR is None:
            IMP_REG_FACTOR=np.zeros(X_train.shape[1])
        if I_constraint is None:
            I_constraint=np.zeros(X_train.shape[1])
        
        BATCH_SIZE = 1           
        criterion = nn.BCEWithLogitsLoss()
        
        X_train = X_train.values
        y_train = y_train.values
        
        train_data = trainData(T.FloatTensor(X_train), T.FloatTensor(y_train))
        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)#########################, shuffle=True)
                
        #criterion = nn.BCELoss()
        # optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE)#, weight_decay=1e-4)#, momentum=momentum)
        
        def binary_acc(y_pred, y_test):
            y_pred_tag = T.round(T.sigmoid(y_pred))
            correct_results_sum = (y_pred_tag == y_test).sum().float()
            acc = correct_results_sum/y_test.shape[0]
            acc = T.round(acc * 100)
            return acc
        
        self.model.train()
        for e in range(1, EPOCHS+1):
            epoch_loss = 0
            epoch_acc = 0
            for en, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                
                y_pred = self.model.predict(X_batch) 
                if sample_weight is not None:
                    criterion = nn.BCEWithLogitsLoss(weight=T.tensor(sample_weight[en])*1000)
                    
                loss = criterion(y_pred, y_batch.unsqueeze(1))

                #-----------------------------------
                # constraints to feature importances
                #-----------------------------------
                layers = [self.model._modules['layer_1']] + [self.model._modules['layer_out']]
                L = len(layers)
        
                # collect the activations
                A = [X_batch[0, :]]+[None]*L
                for l in range(L):
                    if l < (L-1):
                        A[l+1] = T.relu(layers[l].forward(A[l]))
                    else:
                        A[l+1] = layers[l].forward(A[l])

                # compute the relevance score (LRP)                
                R = [None]*L + [(A[-1]).data]
                for l in range(0, L)[::-1]:            
                    w = layers[l].weight
                    #b = layers[l].bias
                    s_j = R[l+1] / ((A[l]*w).t() + 1e-9).sum(0)
                    R[l] = T.matmul((A[l]*w).t(), s_j)
                
                # compute the feature importances
                if R[0].sum() == 0:
                    feature_importances = R[0]
                else:
                    feature_importances = R[0].abs()/(R[0].abs().max())
                
                loss_fol_product_tnorm = T.max(feature_importances - T.Tensor(I_constraint), T.Tensor([0]) ) / (1-T.Tensor(I_constraint)+1e-9)
                loss += T.dot(T.Tensor(IMP_REG_FACTOR), loss_fol_product_tnorm)
                acc = binary_acc(y_pred, y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        
            print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

    def predict(self, X_test):
        X_test = X_test.values
        test_data = testData(T.FloatTensor(X_test))
        test_loader = DataLoader(dataset=test_data, batch_size=100000)
        
        y_pred_train_list = []
        self.model.eval()
        with T.no_grad():
            for X_batch_ in test_loader:
                X_batch_ = X_batch_.to(device)
                y_test_pred = self.model(X_batch_)
                y_test_pred = T.sigmoid(y_test_pred)
                y_pred_train_list.append(y_test_pred.cpu().numpy())
        
        y_pred_train_list = [a.squeeze().tolist() for a in y_pred_train_list]        
        return np.array(y_pred_train_list)[0]

    def importance(self, X, y):
        X = X.values
        y = y.values

        data = trainData(T.FloatTensor(X), T.FloatTensor(y))
    
        layers = [self.model._modules['layer_1']] + [self.model._modules['layer_out']]
        L = len(layers)
        
        data_loader = DataLoader(dataset=data, batch_size=10000, shuffle=True)
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            importances_list = []
            
            for ii in range(X_batch.shape[0]):
                A = [X_batch[ii, :]]+[None]*L
        
                for l in range(L):
                    if l < (L-1):
                        A[l+1] = T.relu(layers[l].forward(A[l]))
                    else:
                        A[l+1] = layers[l].forward(A[l])
                R = [None]*L + [(A[-1]).data]
                
                for l in range(0, L)[::-1]:
                    w = layers[l].weight
                    s_j = R[l+1] / ((A[l]*w).t() + 1e-9).sum(0)
                    R[l] = T.matmul((A[l]*w).t(), s_j)
                
                if R[0].sum() == 0:
                    feature_importances = R[0]
                else:
                    feature_importances = R[0].abs()/(R[0].abs().max())
                    
        
                importances_list.append(T.Tensor.cpu(feature_importances).detach().numpy())
             
        impo = np.mean(np.array(importances_list), 0)
        return impo

