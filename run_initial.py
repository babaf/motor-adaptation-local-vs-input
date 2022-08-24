#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulates a modular RNN model, mimicking motor cortical areas.
The model is trained to produce planar hand position trajectories.
The training/testing data comes from real hand trajectories from two monkeys.
We use pytorch and the Adam optimizer to train the model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time, os
from toolbox_pytorch import RNN, train

starttime = time.time()

rand_seed = 0
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)

# GPU usage #########################################
dtype = torch.FloatTensor # uncomment if you are using CPU
# dtype = torch.cuda.FloatTensor # uncomment if you are using GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DIRECTORY ##########################################
savname = 'results/test/'
if not os.path.exists(savname):
    os.mkdir(savname)

datname = 'data/dataset_chewie_v2_bl'
data = np.load(datname+'.npy',allow_pickle = True).item()
params = data['params']

# PARAMS ###############################################
# target data
output_dim = params['output_dim'] 
target_output = data['target'][:,:,:output_dim] 
# shift such that all have same workspace center
x0 = data['x0']
y0 = data['y0']
target_output[:,:,0] = (target_output[:,:,0] - x0[:,None]) 
target_output[:,:,1] = (target_output[:,:,1] - y0[:,None])

# stimulus parameters from dataset 
tsteps = params['tsteps'] # how many time steps in one trial
ntrials = params['ntrials']
input_dim = params['input_dim']
stimulus = data['stimulus']
dt = params['dt']   

# neuron
n1 = 400 # neuron number PMd
n2 = 400 # neuron number M1
tau = 0.05

# ml params ########################
batch_size = 80
training_trials = 500
lr = 1e-4

alpha1 = 1e-3 # reg inp & out
alpha2 = 1e-3 # reg ff & fb
gamma1 = 1e-3 # reg rec 1
gamma2 = 1e-3 # reg rec 2

beta1 = 0.8 # reg rate 1
beta2 = 0.8 # reg rate 2 

clipgrad = 0.2

params = {
          'n1':n1,'n2':n2,'datname':datname,
          'clipgrad':clipgrad,'beta1':beta1,
          'tau':tau,'dt':dt,'batch_size':batch_size,'rand_seed':rand_seed,
          'training_trials':training_trials,'lr':lr,'alpha1':alpha1,
          'gamma1':gamma1,'alpha2':alpha2,'beta2':beta2,'gamma2':gamma2}


# convert stimulus and target to pytorch form
stim = torch.zeros(training_trials, tsteps, batch_size, input_dim).type(dtype)
target = torch.zeros(training_trials, tsteps, batch_size, output_dim).type(dtype)
for j in range(training_trials):
    idx = np.random.choice(range(ntrials),batch_size,replace=False)
    stimtmp = np.zeros((batch_size,tsteps,input_dim))
    stimtmp[:,:,:2] = stimulus[idx,:,:2]
    stimtmp[:,:,2] = stimulus[idx,:,-1]
    stim[j] = torch.Tensor(stimtmp.transpose(1,0,2)).type(dtype)
    target[j] = torch.Tensor(target_output[idx].transpose(1,0,2)).type(dtype)

# create model
model = RNN(input_dim, output_dim, n1, n2, dt/tau, dtype)
if dtype == torch.cuda.FloatTensor:
    model = model.cuda()
    
# define loss function
criterion = nn.MSELoss(reduction='none') 

# create optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=lr) 

# get and save initial parameters of the model (all are optimized)
params0 = model.save_parameters()

# train it
lc = train(model, optimizer, criterion, training_trials, target, stim,
            alp0=alpha1, bet0=beta1, gam0=gamma1, 
            alp1=alpha2, bet1=beta2, gam1=gamma2, clipv=clipgrad)

# get and save final parameters
params1 = model.save_parameters()

# save it
dic = {'params1':params1,'params0':params0,'lc':np.array(lc),'params':params}
np.save(savname+'training',dic)

# for continuing training later
torch.save({
            'epoch': training_trials,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, savname+'model')

# test it #2
testnames = ['test_set5']

dic = {}
for j,key in enumerate(testnames):
    testdata = data[key]
    stimtmp = np.zeros((testdata['stimulus'].shape[0],tsteps,input_dim))
    stimtmp[:,:,:2] = testdata['stimulus'][:,:,:2]
    stimtmp[:,:,2] = testdata['stimulus'][:,:,-1]
    target_output = testdata['target'][:,:,:output_dim] 
    # shift such that all have same workspace center
    x0 = testdata['x0']
    y0 = testdata['y0']
    target_output[:,:,0] = (target_output[:,:,0] - x0[:,None]) 
    target_output[:,:,1] = (target_output[:,:,1] - y0[:,None])

    stim = torch.Tensor(stimtmp.transpose(1,0,2)).type(dtype)

    # model run
    testout,testl1,testl2,testl1b = model(stim)
    activity1b = testl1b.cpu().detach().numpy().transpose(1,0,2)

    # save it
    output = testout.cpu().detach().numpy().transpose(1,0,2)
    activity1 = testl1.cpu().detach().numpy().transpose(1,0,2)
    activity2 = testl2.cpu().detach().numpy().transpose(1,0,2)

    dic.update({key:{'target':target_output,
                    'cue_onset':testdata['cue_onset'],
    'go_onset':testdata['go_onset'],'trial_id':testdata['trial_id'],
    'activity1':activity1,'activity2':activity2,'output':output,
    'session_id':testdata['session_id'],'activity1b':activity1b}})      

# save it
np.save(savname+'testing',dic)

endtime = time.time()
print('Total time: %.2f'%(endtime-starttime))
