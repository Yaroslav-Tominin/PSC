# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 00:23:38 2022

@author: antoi
"""

from comet_ml import Experiment
import torchaudio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os
import numpy as np
from DCRN128 import DCRN

class Add_noise(nn.Module):
    #Add noise at a random level between 0, -5 and 5 dB
    def __init__(self):
        super(Add_noise,self).__init__()
    def __call__(self, sample):
        signal = sample[0].squeeze(0)
        db = np.random.choice([-5,5])
        mq_s = np.sqrt(torch.mean(signal**2))
        mq_b = np.sqrt(mq_s**2/10**(db/10))
        noise = torch.from_numpy(np.random.normal(0,mq_b,len(signal)))
        #print(noise)
        sn = signal + noise
        return sn[None,:].float()

class STFT(torch.nn.Module):
    def __init__(self):
        super(STFT,self).__init__()
    def __call__(self, sample):
        #print(sample.shape)
        elems = []
        for x in sample:
            spec = torch.stft(x, n_fft = 255, normalized = True).transpose(0,2)
            
            elems.append(spec[None,:])
        res = elems[0]
        for i in range(1,len(elems)):
            res = torch.cat((res,elems[i]),0)
        return res
class ISTFT(torch.nn.Module):
    def __init__(self):
        super(ISTFT,self).__init__()
    def __call__(self, sample):
        #print(sample.shape)
        elems = []
        for x in sample:
            spec = torch.istft(x.squeeze(0), n_fft = 255, normalized =True)
            elems.append(spec[None,:])
        res = elems[0]
        for i in range(1,len(elems)):
            res = torch.cat((res,elems[i]),0)
        return res
noise_audio_transforms = nn.Sequential(
    Add_noise(),
    STFT()

)
clean_audio_transforms = nn.Sequential(
   STFT()
)
"""
import librosa
from matplotlib import pyplot as plt

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)
"""  
def data_processing(data, data_type="train"):
    waves_noise_r = []
    waves_clean_r = []
    waves_noise_c = []
    waves_clean_c = []
    
    waves_noise = []
    waves_clean = []
    for (waveform, _, utterance, _, _, _) in data:
        wave_noise = noise_audio_transforms(waveform).squeeze(0)
        wave_clean = clean_audio_transforms(waveform).squeeze(0)
        #print(wave_noise.shape)
        #plot_spectrogram(spec_noise)
        #plot_spectrogram(spec_clean)
        waves_noise_r.append(wave_noise[0])
        waves_clean_r.append(wave_clean[0])
        waves_noise_c.append(wave_noise[1])
        waves_clean_c.append(wave_clean[1])
        
    waves_noise_r = nn.utils.rnn.pad_sequence(waves_noise_r, batch_first=True).unsqueeze(1)
    waves_clean_r = nn.utils.rnn.pad_sequence(waves_clean_r, batch_first=True).unsqueeze(1)
    waves_noise_c = nn.utils.rnn.pad_sequence(waves_noise_c, batch_first=True).unsqueeze(1)
    waves_clean_c = nn.utils.rnn.pad_sequence(waves_clean_c, batch_first=True).unsqueeze(1)
   
    for i in range(len(waves_noise_r)):
        w_n = torch.cat([waves_noise_r[i],waves_noise_c[i]],0)
        waves_noise.append(w_n)
        w_c = torch.cat([waves_clean_r[i],waves_clean_c[i]],0)
        waves_clean.append(w_c)
        
    waves_noise = torch.stack(waves_noise)
    waves_clean = torch.stack(waves_clean)
    #print(waves_noise.shape)
    return waves_noise, waves_clean
   
   


class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment):
    model.train()
    print("starting training")
    data_len = len(train_loader.dataset)
    
    with experiment.train():
        print("in exp")
        for batch_idx, _data in enumerate(train_loader):
            specs_noise, specs_clean = _data 
            specs_noise, specs_clean = specs_noise.to(device), specs_clean.to(device)
            
            optimizer.zero_grad()
    
            output = model(specs_noise)  # (batch, time, n_class)
            loss = criterion(output, specs_clean)
            loss.backward()
    
            experiment.log_metric('loss', loss.item(), step=iter_meter.get())
            experiment.log_metric('learning_rate', scheduler.get_lr(), step=iter_meter.get())
    
            optimizer.step()
            scheduler.step()
            iter_meter.step()
            
            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(specs_noise), data_len,
                    100. * batch_idx / len(train_loader), loss.item()))
    
    
def test(model, device, test_loader, criterion, epoch, iter_meter, experiment):
    print('\nevaluating…')
    model.eval()
    test_loss = 0
    with experiment.test():
        with torch.no_grad():
            for I, _data in enumerate(test_loader):
               specs_noise, specs_clean = _data 
               specs_noise, specs_clean = specs_noise.to(device), specs_clean.to(device)
    
               output = model(specs_noise)  # (batch, time, n_class)
              
               loss = criterion(output, specs_clean)
               test_loss += loss.item() / len(test_loader)
               
               

   
    experiment.log_metric('test_loss', test_loss, step=iter_meter.get())
  
    print('Test set: Average loss: {:.4f}\n'.format(test_loss))

def main(experiment,learning_rate=5e-3, batch_size=32, epochs=1,
    train_url="train-clean-100", test_url="test-clean"):
    
    hparams = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }
    standard_enc = {"fbins" : [128,64,32,16,8,4,2,1], "channels" : [2,32,32,32,32,64,128,256,512]}
    standard_dec = {"fbins" : [2,4,8,16,32,64,128,256], "channels" : [256,128,64,32,32,32,32,2]}
        

    experiment.log_parameters(hparams)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")
    model = DCRN(standard_enc,standard_dec)
    model.to(device)
    
    #print("saved")
    
    x = torch.randn((16,2,40,128)).to(device)
    print(type(x))
    out = model(x)
    print(type(out))
    
    train_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=train_url, download = True)
    test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=test_url, download = True)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, 'train'),
                                **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'valid'),
                                **kwargs)

    
    
    #print(next(iter(train_loader))[0][0][0].shape)
    #print(plot_spectrogram(next(iter(train_loader))[0][0][0]))
    #print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
                                            steps_per_epoch=int(len(train_loader)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy='linear')

    iter_meter = IterMeter()
    
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment)
        test(model, device, test_loader, criterion, epoch, iter_meter, experiment)
    torch.save(model.state_dict(), "dcrn.pt")
    
if __name__ == "__main__":
    print("starting script")
    
    experiment = Experiment(
    api_key="3R2GzsUplN6iNSQJFeYBO0gD4",
    project_name="le-psc",
    workspace="antoinemsy",
)
    
    print("experiment loaded")
    main(experiment)