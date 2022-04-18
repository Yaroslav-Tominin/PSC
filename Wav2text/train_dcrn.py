# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:21:45 2022

@author: antoi
"""
from DCRN import DCRN

import torch as t
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

local_path = r"C:\Users\antoi\Documents\X\PSC\PSCcode\data.json"
path = "data.json"
class CustomDataset(t.utils.data.Dataset):
    def __init__(self,path):
        super(CustomDataset, self).__init__()
        self.dic = pd.read_json(path)
        print(self.dic.head)
        for i in range(len(self.dic)):
            t0 = t.tensor(self.dic["data"][i][0])
            t1 = t.tensor(self.dic["data"][i][1])
            self.dic["data"][i] = (t0,t1)
        self.num_items = self.dic.size

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        item = self.dic["data"][index]
        return (item[0],item[1])



def test():
    model = DCRN()
    x = t.randn((batch_size,2,40,256))
    out = model.forward(x)
    
    print(type(out))
    print(out.shape)
    


#hyperparameters
Learning_rate = 1e-4
device = "cuda" if t.cuda.is_available() else "cpu"
batch_size = 16
num_epochs = 1

#Data loading
dataset = CustomDataset(path)
print("loading data")
train_loader = t.utils.data.DataLoader(dataset, batch_size = 16, shuffle = True)

#Entraînement du modèle
import torch.optim as optim
model = DCRN()
loss = t.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train(loader, model,optimizer, loss_fn):
    #loop = tqdm(loader)
    print("starting training ...")
    for batch_idx, (data,targets) in tqdm(enumerate(loader)):
        print(batch_idx)
        optimizer.zero_grad()
        data = t.stft(data,511, hop_length = 1)
        data = data.reshape(1,2,-1,256)
        data = t.tensor(data).to(device = device)
        targets = t.tensor(targets).to(device = device)
        #forward
        with t.cuda.amp.autocast():
            predictions = model(data)
            predictions = t.istft(predictions,511)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
        print(loss)
        
    print("training finished")
if __name__ == '__main__':
    test()
    train(train_loader,model,optimizer, loss)
