#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

# seeds are set so reproducible results can be obtained
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False

import torchvision
from torch.utils.data import Dataset, DataLoader             # Classes
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

import numpy as np
import math

from sklearn import datasets                                 # to have a binary classification dataset
from sklearn.preprocessing import StandardScaler             # to scale features
from sklearn.model_selection import train_test_split         # seperation of training and testing data
from sklearn.metrics import classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(0)


# In[2]:


inp = input("FC or CNN")


# In[3]:


class BreathDatasetTrain(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('./JupyterProjects/B_Ing/Honours/example_train.csv', delimiter=",", dtype=np.float32)  # delimiter: , seperated file  --  skip first row: header
        self.features = torch.from_numpy(xy[:, 1:]).to(device)
        self.labels = torch.from_numpy(xy[:, [0]]).type(torch.LongTensor).to(device) # n_samples, 1
        self.n_samples = xy.shape[0]
        
        
    def __getitem__(self, index):
        # to call an index in the dataset: dataset[0] for example
        return self.features[index], self.labels[index]   # --> tuple
    
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
    
class BreathDatasetVal(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('./JupyterProjects/B_Ing/Honours/example_val.csv', delimiter=",", dtype=np.float32)  # delimiter: , seperated file  --  skip first row: header
        self.features = torch.from_numpy(xy[:, 1:]).to(device)
        self.labels = torch.from_numpy(xy[:, [0]]).to(device) # n_samples, 1
        self.n_samples = xy.shape[0]
        
        
    def __getitem__(self, index):
        # to call an index in the dataset: dataset[0] for example
        return self.features[index], self.labels[index]   # --> tuple
    
    def __len__(self):
        # len(dataset)
        return self.n_samples

    
class BreathDatasetTest(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('./JupyterProjects/B_Ing/Honours/example_test.csv', delimiter=",", dtype=np.float32)  # delimiter: , seperated file  --  skip first row: header
        self.features = torch.from_numpy(xy[:, 1:]).to(device)
        self.labels = torch.from_numpy(xy[:, [0]]).to(device) # n_samples, 1
        self.n_samples = xy.shape[0]
        
        
    def __getitem__(self, index):
        # to call an index in the dataset: dataset[0] for example
        return self.features[index], self.labels[index]   # --> tuple
    
    def __len__(self):
        # len(dataset)
        return self.n_samples


# In[4]:


class CONVNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CONVNeuralNet, self).__init__()
        stride = 3
        
        self.conv1 = nn.Conv1d(1, 1, 20)
        self.conv2 = nn.Conv1d(1, 1, 5)
        self.conv3 = nn.Conv1d(1, 1, 4)
        self.conv4 = nn.Conv1d(1, 1, 3)
        
        self.pool = nn.MaxPool1d(3, stride=stride)           # kernel size, stride (=shift to the right)
        
        self.l1 = nn.Linear(109, 50)
        self.l2 = nn.Linear(50, num_classes)
        
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        # print(x.shape)
        out = self.conv1(x)
        # print(out.shape)
        out = self.relu(out)
        # print(out.shape)
        out = self.pool(out)
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = self.relu(out)
        # print(out.shape)
        out = self.pool(out)
        out = out.squeeze(dim=1)
        # print(out.shape)
        
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        # no softmax because in cross entropy function
        return out


# In[5]:


class FCNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FCNeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, 30)
        # self.l3 = nn.Linear(40, 20)
        self.l4 = nn.Linear(30, num_classes)
        # self.l5 = nn.Linear(20, num_classes)
        
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        # print(x.shape)
        out = self.l1(x)
        # print(out.shape)
        out = self.relu(out)
        # print(out.shape)
        out = self.l2(out)
        # print(out.shape)
        out = self.relu(out)
        # print(out.shape)
        # out = self.l3(out)
        # print(out.shape)
        # out = self.relu(out)
        # print(out.shape)
        out = self.l4(out)
        # no softmax because in cross entropy function
        return out


# In[6]:


def getData(batch_size):
    torch.manual_seed(0)

    dataloader_train = DataLoader(dataset=BreathDatasetTrain(), batch_size=batch_size, shuffle=True)    # num_workers can make it faster: uses multiple subprocesses
    dataloader_val = DataLoader(dataset=BreathDatasetVal(), batch_size=batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset=BreathDatasetTest(), batch_size=batch_size, shuffle=False)
    
    # print("----------------- getData success ----------------")
    return dataloader_train, dataloader_val, dataloader_test


# In[7]:


def trainModel(num_epochs, dataloader_train, dataloader_val, model, criterion, optimizer, writer, final=0):
    for epoch in range(num_epochs):
        # if (epoch+1) % 20 == 0:
        #        print(f"---- epoch = {epoch+1}/{num_epochs} ----")
        for i, (inputs, labels) in enumerate(dataloader_train):
            #forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # if (i+1) % 100 == 0:
                # print(f"epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}")
            
        if final:
            evaluateModel(dataloader_val, model, writer, final)
    
    print(f"Final loss = {round(loss.item(), 6)}\n")
    return evaluateModel(dataloader_val, model, writer, 1)
    # print("----------------- trainModel success -----------------")


# In[8]:


def evaluateModel(dataloader, model, writer, final=0):
    with torch.no_grad():
        predictions = []
        labels = []
        for inputs, labels_per_batch in dataloader:
            outputs = model(inputs)
            _, predictions_per_batch = torch.max(outputs, 1)
            predictions += (predictions_per_batch)
            labels += labels_per_batch.squeeze()
    
        predictions = torch.tensor(predictions)
        labels = torch.tensor(labels)
    
        if final:
            print(classification_report(labels, predictions, zero_division=0, target_names=['Eupnea', 'Apnea', 'Kussmaul', 'Cheyne-Stokes']))
            return classification_report(labels, predictions, zero_division=0, output_dict=True)['accuracy']


# In[9]:


def saveM(model, batch_size, hidden_size, learning_rate, index):
    if inp == "FC":
        FILE = './JupyterProjects/B_Ing/Honours/Models/FC1/model' + str(index) + '.pth'
    else:
        FILE = './JupyterProjects/B_Ing/Honours/Models/Conv1/model' + str(index) + '.pth'
    savePoint = {
        'model': model.state_dict(),
        'batch_size': batch_size,
        'hidden_size': hidden_size,
        'learning_rate': learning_rate
    }
    
    torch.save(savePoint, FILE)
    
    print("Saved at", FILE)
    
def loadM(index):
    if inp == "FC":
        FILE = './JupyterProjects/B_Ing/Honours/Models/FC1/model' + str(index) + '.pth'
    else:
        FILE = './JupyterProjects/B_Ing/Honours/Models/Conv1/model' + str(index) + '.pth'
    
    return torch.load(FILE)


# In[10]:


def runner(good_models, batch_size, learning_rate, hidden_size, input_size, num_classes, num_epochs):
    if inp == "CNN":
        hidden_size = 0
        dataloader_train, dataloader_val, dataloader_test = getData(batch_size)

        writer = SummaryWriter("runs/FC_class/bs" + str(batch_size) + "_hs" + str(hidden_size) + "_lr" + str(learning_rate))
        torch.manual_seed(0)
        model = CONVNeuralNet(input_size, hidden_size, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        good_model = trainModel(num_epochs, dataloader_train, dataloader_val, model, criterion, optimizer, writer)
        if good_model == 1.0:
            good_models += 1

            saveM(model, batch_size, hidden_size, learning_rate, good_models)
    else:
        dataloader_train, dataloader_val, dataloader_test = getData(batch_size)

        writer = SummaryWriter("runs/FC_class/bs" + str(batch_size) + "_hs" + str(hidden_size) + "_lr" + str(learning_rate))
        torch.manual_seed(0)
        model = FCNeuralNet(input_size, hidden_size, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        good_model = trainModel(num_epochs, dataloader_train, dataloader_val, model, criterion, optimizer, writer)
        if good_model == 1.0:
            good_models += 1

            saveM(model, batch_size, hidden_size, learning_rate, good_models)
    
    return good_models


# In[11]:


def main():    
    bd_train = BreathDatasetTrain()                         # Consists of 8 000 training samples
    bd_val = BreathDatasetVal()                             # Consists of 1 000 validation samples
    bd_test = BreathDatasetTest()                           # Consists of 1 000 testing samples
    
    good_models = 0
    
    batch_sizes = [50, 100, 200]
    learning_rates = [0.5, 0.3, 0.1, 0.01, 0.001] 
    hidden_sizes = [50, 200, 400, 600, 800]
    
    num_training_samples = len(bd_train)
    num_classes = 4
    num_epochs = 10
    input_size = 1020
    
    run = 0
    
    for batch_size in batch_sizes:        
        for learning_rate in learning_rates:
            if inp == "CNN":
                hidden_size = 0
                
                run += 1
                print(f"------------------------------- run {run} -------------------------------")
                print(f"Batch size = {batch_size}, learning_rate = {learning_rate}, hidden_size = {hidden_size}")

                good_models = runner(good_models, batch_size, learning_rate, hidden_size, input_size, num_classes, num_epochs)
            else:
                for hidden_size in hidden_sizes:
                    # To keep track of model used (hyperparameters) and progress
                    run += 1
                    print(f"------------------------------- run {run} -------------------------------")
                    print(f"Batch size = {batch_size}, learning_rate = {learning_rate}, hidden_size = {hidden_size}")

                    good_models = runner(good_models, batch_size, learning_rate, hidden_size, input_size, num_classes, num_epochs)
        
    return good_models


# In[12]:


def test(good_models):
# Test best hyperparameters
    print("\n"*3 + "-"*60 +  " TEST " + "-"*60 + "\n"*3)
    test_run = 0
    
    input_size = 1020
    num_classes = 4
    
    for index in range(good_models):
        test_run += 1
        loaded_dict = loadM(test_run)
        hidden_size = loaded_dict['hidden_size']
        if inp == "CNN":
            model = CONVNeuralNet(input_size, hidden_size, num_classes)
        else: 
            model = FCNeuralNet(input_size, hidden_size, num_classes)
        model.load_state_dict(loaded_dict['model'])
        model.to(device).eval()
        batch_size = loaded_dict['batch_size']
        learning_rate = loaded_dict['learning_rate']
        
        
        print(f"------------------------------- test run {test_run} -------------------------------")
        print(f"Batch size = {batch_size}, learning_rate = {learning_rate}, hidden_size = {hidden_size}")
        
        dataloader_train, dataloader_val, dataloader_test = getData(batch_size)

        writer = SummaryWriter("runs/FC_class/testrun")        
        evaluateModel(dataloader_test, model, writer, 1)


# In[13]:


good_models = main()


# In[14]:


test(good_models)


# In[ ]:




