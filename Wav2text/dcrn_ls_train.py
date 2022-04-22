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
        print(sample.shape)
        signal = sample[0].squeeze(0)
        db = np.random.choice([15])
        mq_s = np.sqrt(torch.mean(signal**2))
        mq_b = np.sqrt(mq_s**2/10**(db/10))
        noise = torch.from_numpy(np.random.normal(0,mq_b,len(signal)))
        #print(noise)
        sn = signal + noise
        return sn[None,:].float()
    
def loop_audio(source,length,desired_length):
    res = source
    if desired_length>length:
        iters = desired_length//length
        remaining = desired_length-length*iters
        for i in range(1,iters):
            res = torch.cat((res,source),1)
        last = source[:, :remaining]
        res= torch.cat((res,last),1)
    else:
        res = source[:,:desired_length]
    return res


def add_real_noise(waveform,snr_db = 20):
    audio_noise = "Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav"
    
    noise,sr= torchaudio.load(audio_noise)
    
    wave_power = waveform.norm(p=2)
    noise_power = noise.norm(p=2)
    
    noise = loop_audio(noise,noise.shape[1],waveform.shape[1])
   
    snr = 10 ** (snr_db / 20)
    scale = snr * noise_power / wave_power
    return (scale * waveform + noise) / 2*5

class Add_real_noise(nn.Module):
    def __init__(self):
        super(Add_real_noise,self).__init__()
    def __call__(self, sample):
        return add_real_noise(sample)
    
"""
OLD Transforms
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
"""
def spectro(x, n_fft=511, hop_length=None, pad=0):
    *other, length = x.shape
    x = x.reshape(-1, length)
    z = torch.stft(x,
                n_fft * (1 + pad),
                hop_length or n_fft // 4,
                window=torch.hann_window(n_fft).to(x),
                win_length=n_fft,
                normalized=True,
                center=True,
                pad_mode='reflect')#(batch,freqs,frames,channels)
    
    return z.transpose(1,3)#(batch,channels,frames,freqs)


def ispectro(z, n_fft = 511,hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    #n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    x = torch.istft(z,
                 n_fft,
                 hop_length,
                 window=torch.hann_window(win_length).to(z),
                 win_length=win_length,
                 normalized=True,
                 length=length,
                 center=True)
    #_, length = x.shape
    return x[None,:]

class STFT(torch.nn.Module):
    def __init__(self):
        super(STFT,self).__init__()
    def __call__(self, sample):
        return spectro(sample)

class ISTFT(torch.nn.Module):
    def __init__(self):
        super(STFT,self).__init__()
    def __call__(self, sample):
        return ispectro(sample)
noise_audio_transforms = nn.Sequential(
    Add_real_noise(),
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
    return waves_noise, waves_clean
   
   


class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val

def custom_loss(pred,target):
    """computes L1 loss separating real and imaginary parts"""
    l1_loss = torch.nn.L1Loss()
    real_p,real_t = pred[:,0,:,:],target[:,0,:,:]
    comp_p,comp_t = pred[:,1,:,:],target[:,1,:,:]
    loss = l1_loss(real_p,real_t)+l1_loss(comp_p,comp_t)
    return loss

def train(model, device, train_loader, optimizer, scheduler, epoch, iter_meter, experiment):
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
            loss = custom_loss(output, specs_clean)
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
    
    
def test(model, device, test_loader, epoch, iter_meter, experiment):
    print('\nevaluating…')
    model.eval()
    test_loss = 0
    
    with experiment.test():
        with torch.no_grad():
            for I, _data in enumerate(test_loader):
               specs_noise, specs_clean = _data 
               specs_noise, specs_clean = specs_noise.to(device), specs_clean.to(device)
    
               output = model(specs_noise)  # (batch, time, n_class)
              
               loss = custom_loss(output, specs_clean)
               
               test_loss += loss.item() / len(test_loader)
               
               

   
    experiment.log_metric('test_loss', test_loss, step=iter_meter.get())
  
    print('Test set: Average loss: {:.4f}\n'.format(test_loss))

def main(experiment,learning_rate=5e-4, batch_size=8, epochs=1,
    train_url="train-clean-100", test_url="test-clean"):
    
    hparams = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }
    #standard_enc = {"fbins" : [256,128,64,32,16,8,4,2,1], "channels" : [2,32,32,32,32,64,128,256,512]}
    #standard_dec = {"fbins" : [2,4,8,16,32,64,128,256,256], "channels" : [256,128,64,32,32,32,32,2]}
        

    experiment.log_parameters(hparams)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")
    model = DCRN()
    model.to(device)
    
    #print("saved")
    
   
    
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
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 500, gamma = 0.4)

    iter_meter = IterMeter()
    
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, scheduler, epoch, iter_meter, experiment)
        torch.save(model.state_dict(), "dcrn.pt")
        test(model, device, test_loader, epoch, iter_meter, experiment)
   
    
if __name__ == "__main__":
    print("starting script")
    
    experiment = Experiment(
    api_key="3R2GzsUplN6iNSQJFeYBO0gD4",
    project_name="le-psc",
    workspace="antoinemsy",
)
    
    print("experiment loaded")
    main(experiment)