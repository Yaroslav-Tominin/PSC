#TEST MODEL
import torch as t
import torch.nn as nn

import json
a_dict = json.load(open("test.json"))


test_audio = t.tensor(a_dict["common_voice_br_17977507.mp3"])


class EConv(nn.Module):
    #Conv followed by batch norm and ReLU
    def __init__(self, 
                 input_channels, 
                 output_channels,
                 fbins, 
                 time_dim = 40,
                 kernel_size = 3,
                 stride=(1,2)):
        super(EConv,self).__init__()
    
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

class DConv(nn.Module):
    #Convolution (ideally subpixelconvolution, but here just classic convo) + Batch Norm + ReLU
    def __init__(self,
                 input_channels,
                 output_channels,
                 fbins,
                 kernel_size = 3,
                 stride=(1,1)):
        super(DConv,self).__init__()
        self.stride = stride
        if kernel_size == 3 : 
            padding = (1,1)
        else : 
            padding = (2,2)
        self.conv = nn.Conv2d(in_channels=input_channels,
                              kernel_size = kernel_size,
                              
                              out_channels= output_channels, stride=self.stride, padding = padding)
        self.ln = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        
    def forward(self,batch_data):
        batch_data.float()
        output = self.conv(batch_data)
        output = self.ln(output)
        output = self.relu(output)
        return output
    
   
class doubleBLSTM(nn.Module):
    def __init__(self):
        super(doubleBLSTM,self).__init__()
        self.hidden_size = 256
        self.lstm = nn.LSTM(input_size = 512,  hidden_size = self.hidden_size, num_layers = 2,
                               bidirectional= True, batch_first=True)
        
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
       
    def forward(self, batch_data):
        print(batch_data.shape)
        batch_data = batch_data.squeeze(3).transpose(1,2)
        print(batch_data.shape)
        output , _ = self.lstm(batch_data)
        output = output[None,:].transpose(0,1).transpose(1,3)
        print(output.shape)
        return output
    
class DCRN(nn.Module):
    def __init__(self, fbins = 128,
    input_channels = 2,
    output_channels = 32,
    dense_depth = 5,
    additionnal_depth = 3):
        
        super(DCRN,self).__init__()
        
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        #ENCODER
        self.encoder = []
        self.fbins = fbins*2
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.e_depth = dense_depth + additionnal_depth
        
        #SKIPPING DENSE BLOCKS FIRST
        
        for i in range(dense_depth-1):
            self.cell = EConv(self.input_channels, self.output_channels, fbins = self.fbins)
            #self.dense = E_Dense_Block(self.output_channels,self.output_channels)
            self.fbins //=2
            self.input_channels = self.output_channels
            self.encoder.append(nn.Sequential(self.cell))
            
        #Additional layer with doubling output_channels number
        self.output_channels *=2
        self.cell = EConv(self.input_channels, self.output_channels, fbins = self.fbins)
        self.fbins//=2
        #self.dense = E_Dense_Block(self.output_channels,self.output_channels)
        self.input_channels = self.output_channels
        self.encoder.append(nn.Sequential(self.cell))
        
        #Layers without dense blocks
        for i in range(additionnal_depth):
            self.output_channels *=2
            self.cell = EConv(self.input_channels, self.output_channels, fbins = self.fbins)
            self.fbins //=2
            
            self.input_channels = self.output_channels
            self.encoder.append(nn.Sequential(self.cell))
            
        #2 BILSTM
        self.lstm = doubleBLSTM()
       
        #DECODER
        self.decoder = []
        #1 concatenation
        self.fbins=1
        
        #2 Layers without dense block
        for i in range(additionnal_depth):
            self.output_channels //=2
            self.fbins *=2
            self.cell = DConv(self.input_channels, self.output_channels,fbins = self.fbins)
          
            self.input_channels = self.output_channels
            self.decoder.append(nn.Sequential(self.cell))
            
        #SKIPPING DENSE BLOCKS
        for i in range(dense_depth-1):
            self.output_channels = 32
            self.fbins *=2
            self.cell = DConv(self.input_channels, self.output_channels,fbins = self.fbins)
    
            #self.dense = E_Dense_Block(self.output_channels,self.output_channels)
            
            self.input_channels = self.output_channels
            self.decoder.append(nn.Sequential(self.cell))
            
        #1 last conv, no concatenation, 2 output channels, 256 fbins output
        self.lastcell = nn.Sequential(DConv(self.input_channels, 2,256))
        
    def forward(self,batch_data):
        y = t.tensor(batch_data).to(self.device)
        saved = []
        ln = 0
        for x in self.encoder:
            y = x(y)
            saved.append(y)
        y = self.lstm(y)
        for x in self.decoder:
            e_con = saved.pop()
            ln+=1
            y = t.cat((y,e_con), 3)
            y = x(y)
            
        e_con = saved.pop()
        y = t.cat((y,e_con), 3)
        y = self.lastcell(y)   
        return y
    
"""
def test():
    model = DCRN()
    x = t.randn((16,2,40,256))
    out = model.forward(x)
    
    print(type(out))
    print(out.shape)

test()
"""