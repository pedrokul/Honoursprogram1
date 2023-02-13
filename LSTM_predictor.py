#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from torch.utils.data import Dataset, DataLoader             # Classes
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import math

import matplotlib.pyplot as plt

from sklearn import datasets                                 # to have a binary classification dataset
from sklearn.preprocessing import StandardScaler             # to scale features
from sklearn.model_selection import train_test_split         # seperation of training and testing data
from sklearn.metrics import classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(0)

# prediction gives on how muwh values the prediction is based
prediction = 17


# In[2]:


def getData(batch_size):
    torch.manual_seed(0)

    dataloader_train = DataLoader(dataset=BreathDatasetTrain(), batch_size=batch_size, shuffle=True)    # num_workers can make it faster: uses multiple subprocesses
    dataloader_val = DataLoader(dataset=BreathDatasetVal(), batch_size=batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset=BreathDatasetTest(), batch_size=batch_size, shuffle=False)
    
    # print("----------------- getData success ----------------")
    return dataloader_train, dataloader_val, dataloader_test


# In[3]:


class BreathDatasetTrain(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('./JupyterProjects/B_Ing/Honours/example_train.csv', delimiter=",", dtype=np.float32)  # delimiter: , seperated file  --  skip first row: header
        self.labels = torch.from_numpy(xy[:, [0]]).to(device)
        self.features = torch.from_numpy(xy[:, 1:-prediction]).to(device)
        self.targets = torch.from_numpy(xy[:, prediction+1:]).to(device) # n_samples, 1
        self.n_samples = xy.shape[0]
        
        
    def __getitem__(self, index):
        # to call an index in the dataset: dataset[0] for example
        return self.features[index], self.targets[index]   # --> tuple
    
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
    
class BreathDatasetVal(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('./JupyterProjects/B_Ing/Honours/example_val.csv', delimiter=",", dtype=np.float32)  # delimiter: , seperated file  --  skip first row: header
        self.labels = torch.from_numpy(xy[:, [0]]).to(device)
        self.features = torch.from_numpy(xy[:, 1:-prediction]).to(device)
        self.targets = torch.from_numpy(xy[:, prediction+1:]).to(device) # n_samples, 1
        self.n_samples = xy.shape[0]
        
        
        
    def __getitem__(self, index):
        # to call an index in the dataset: dataset[0] for example
        return self.features[index], self.targets[index]   # --> tuple
    
    def __len__(self):
        # len(dataset)
        return self.n_samples

    
class BreathDatasetTest(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('./JupyterProjects/B_Ing/Honours/example_test.csv', delimiter=",", dtype=np.float32)  # delimiter: , seperated file  --  skip first row: header
        self.labels = torch.from_numpy(xy[:, [0]]).to(device)
        self.features = torch.from_numpy(xy[:, 1:-prediction]).to(device)
        self.targets = torch.from_numpy(xy[:, prediction+1:]).to(device) # n_samples, 1
        self.n_samples = xy.shape[0]
        
        
        
    def __getitem__(self, index):
        # to call an index in the dataset: dataset[0] for example
        return self.labels[index], self.features[index], self.targets[index]   # --> tuple
    
    def __len__(self):
        # len(dataset)
        return self.n_samples


# In[4]:


bd = BreathDatasetTest()
label, features, targets = bd[0]
print(label)


# In[5]:


class LSTMPredictor(nn.Module):
    def __init__(self, hidden_size=50):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size 
        self.lstm1 = nn.LSTMCell(prediction, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, prediction)
        
    def forward(self, x, future=0):
        outputs = []
        num_samples = x.size(0)
        
        h_t1 = torch.zeros(num_samples, self.hidden_size, dtype=torch.float32).to(device)
        c_t1 = torch.zeros(num_samples, self.hidden_size, dtype=torch.float32).to(device)
        h_t2 = torch.zeros(num_samples, self.hidden_size, dtype=torch.float32).to(device)
        c_t2 = torch.zeros(num_samples, self.hidden_size, dtype=torch.float32).to(device)
        h_t3 = torch.zeros(num_samples, self.hidden_size, dtype=torch.float32).to(device)
        c_t3 = torch.zeros(num_samples, self.hidden_size, dtype=torch.float32).to(device)
        
        #splits the input into chunks of length pedriction (=17)
        for input_t in x.split(prediction, dim=1):
            h_t1, c_t1 = self.lstm1(input_t, (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.linear(h_t3)
            outputs.append(output)
            
        # calculates how many chunks there are in the to be predicted vector, only is used when future != 0
        next = int(future/prediction)
        # calculates future chunks
        for i in range(next):
            h_t1, c_t1 = self.lstm1(output, (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.linear(h_t3)
            outputs.append(output)
        
        # makes a 1D vector of outputs
        outputs = torch.cat(outputs, dim=1).to(device)
        return outputs


# In[6]:


def trainModel(dataloader_train, model, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(dataloader_train):
            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if (i+1) % 20 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, step {i+1}/{len(dataloader_train)}, loss = {loss.item():.4f}") 
                loss.backward()
                return loss

            optimizer.step(closure)
            


# In[7]:


def evaluateModel(dataloader, model, criterion):
    with torch.no_grad():
        future = 17*60
        for labels, inputs, targets in dataloader:
            # predictions are calculated of the original time sequence and in addition 1020 future values are predicted
            preds = model(inputs, future)
            # print(inputs.size(), targets.size(), preds.size())
            # loss of the prediction untill future predictions is calculated
            loss = criterion(preds[:, :-future], targets)
            print(f"Loss: {loss.item()}")
            y = preds.detach().cpu().numpy()

            # predictions are drawn in full red and future is drawn in pointed red
            input_size = 1020
            for i in range(1):
                draw(y[i], targets[i], input_size, input_size, 'r', labels[i].item())
                print(y[i])
        return y


# In[8]:


def draw(y, target, input_size, future, color, label):
    x = np.linspace(0, 120, 120*17)
    targets = target.detach().cpu().numpy()
    plt.figure(figsize=(14,8))
    if label == 0:
        plt.title('Eupnea', fontsize=20)
    elif label == 1:
        plt.title('Apnea', fontsize=20)
    elif label == 2:
        plt.title('Kussmaul', fontsize=20)
    else:
        plt.title('Cheyne-Stokes', fontsize=20)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(x[prediction:input_size], targets, "b:", linewidth=1.0)
    plt.plot(x[prediction:input_size], y[:input_size-prediction], color, linewidth=2.0)
    plt.plot(x[input_size:], y[input_size-prediction:], color + ":", linewidth=2.0)
    plt.show()


# In[9]:


def model():
    bd_train = BreathDatasetTrain()
    batch_size = 50
    learning_rate = 0.1
    
    num_training_samples = len(bd_train)
    num_epochs = 1
    hidden_size = 100
    input_size = 1020

    dataloader_train, dataloader_val, dataloader_test = getData(batch_size)
                        
    torch.manual_seed(0)        
    model = LSTMPredictor(hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
    
    trainModel(dataloader_train, model, optimizer, criterion, num_epochs)
    
    evaluateModel(dataloader_test, model, criterion)


# In[10]:


model()


# In[ ]:




