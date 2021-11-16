#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perturbation experiment for initially trained network.
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
perturbation = 'vmr'
subname = 'null' # select the plasticity type ['extrainput','noupstream','null']

# load data from initial training
trainingdata = np.load(savname+'training.npy',allow_pickle=True).item()
trainingparams = trainingdata['params']
datname = trainingparams['datname']

# create new subdirectory with perturbation experiment
savname2 = savname + subname + '/'
if not os.path.exists(savname2):
    os.mkdir(savname2)

# load training data for perturbation experiment
datname = datname+perturbation
data = np.load(datname+'.npy',allow_pickle = True).item()
params = data['params']

# PARAMS ##############################################
whichplastic = subname
training_trials = 0 if subname=='null' else 100

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
trial_id = data['trial_id']
input_dim = params['input_dim']
stimulus = data['stimulus']
dt = trainingparams['dt']
# neuron
n1 = trainingparams['n1'] # neuron number PMd
n2 = trainingparams['n2'] # neuron number M1
tau = trainingparams['tau']
# ml params
batch_size = 1
lr = trainingparams['lr']
alpha1 = trainingparams['alpha1']
alpha2 = trainingparams['alpha2']
beta1 = trainingparams['beta1']
beta2 = trainingparams['beta2']
gamma1 = trainingparams['gamma1']
gamma2 = trainingparams['gamma2']
clipgrad = trainingparams['clipgrad']

# SETUP ################################################
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

# fix output weights
model.output.weight.requires_grad = False 
model.output.bias.requires_grad = False 

if whichplastic == 'extrainput':
    model.rnn_l0.weight_hh_l0.requires_grad = False # recurrent weights
    model.rnn_l0.weight_ih_l0.requires_grad = False
    model.rnn_l1.weight_hh_l0.requires_grad = False # recurrent weights
    model.rnn_l1.weight_ih_l0.requires_grad = False 
elif whichplastic == 'noupstream':
    model.rnn_l0b.weight_hh_l0.requires_grad = False
    model.rnn_l0b.weight_ih_l0.requires_grad = False
    model.rnn0bto0.weight.requires_grad = False
elif whichplastic == 'allplastic' or whichplastic == 'all' or whichplastic == 'null' or whichplastic == 'null':
    pass
else:
    print('ERROR: whichplastic not known')

# create optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=lr) 

# load
initial_training = torch.load(savname+'model')
model.load_state_dict(initial_training['model_state_dict'])

# get and save initial parameters
params1 = model.save_parameters()

# train it
lc = train(model, optimizer, criterion, training_trials, target, stim, 
           alp0=alpha1, bet0=beta1, gam0=gamma1, 
           alp1=alpha2, bet1=beta2, gam1=gamma2, clipv=clipgrad)

# get and save final parameters
params2 = model.save_parameters()

# save it
dic = {'params1':params1,'params2':params2,'lc':np.array(lc)}
np.save(savname2+'training_'+perturbation,dic)

# for continuing training later
torch.save({
            'epoch': training_trials,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, savname2+'model_'+perturbation)

# test it
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
np.save(savname2+'testing_'+perturbation,dic)

endtime = time.time()
print('Total time: %.2f'%(endtime-starttime))
