# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:48:58 2022

@author: antoi
"""
import torch
import torch.nn as nn

class Conv(nn.Module):
    #Conv followed by batch norm and ReLU
    def __init__(self, 
                 input_channels, 
                 output_channels,
                 fbins, 
                 padding = (1,1),
                 kernel_size = 3,
                 stride=(1,1)):
        super(Conv,self).__init__()
    
        self.conv = nn.Conv2d(in_channels=input_channels,
                              kernel_size = kernel_size,
                              
                              out_channels= output_channels, stride=stride, padding = (1,1))
        self.ln = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        
    def forward(self,batch_data):
        output = self.conv(batch_data)
        output = self.ln(output)
        output = self.relu(output)
        return output
    
class CNN_test(nn.Module):
    def __init__(self):
        super(CNN_test,self).__init__()
        self.cell1 = Conv(2,32,128, kernel_size=5, padding = (2,2))
        self.cell2 = Conv(32,32,128)
        self.cell3 = Conv(32,64,128)
        self.cell4 = Conv(64,32,128)
        self.cell5 = Conv(32,2,128)
        
        self.model = nn.Sequential(self.cell1,self.cell2,self.cell3,self.cell4,self.cell5)
    
    def forward(self,x):
        return self.model(x)

def test():
    model = CNN_test()
    x = torch.randn((16,2,40,128))
    #params = model.parameters()
    #print(type(model(x)))
    #model.cuda()
    #x = x.cuda()
    #print(type(model(x)))
    out = model.forward(x)
   
    print(type(out))
    print(out.shape)
    
if __name__ == "__main__":
    test()
    