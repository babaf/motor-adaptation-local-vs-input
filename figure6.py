"""
Look at robustness to synaptic noise.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (2,2)
mpl.rcParams['figure.dpi'] = 96
col = {'pmd':'#aade87','m1':'#008000'}


def calc_dw(dinit):
    try:
        dif = []
        dim = []
        for k in code_order:
            w = dinit['params1'][k]
            w2 = dinit['params2'][k]
            dif.append(np.median(abs((w2-w)/(w)*100)))
            try:   
                _,e,_ = np.linalg.svd(w2-w)
                pr = np.sum(e.real)**2/np.sum(e.real**2)
                dim.append(pr)
            except:
                dim.append(np.nan)
    except:
        dif = []
        dim = []
        for k in code_order:
            w = dinit['params0'][k]
            w2 = dinit['params1'][k]
            dif.append(np.median(abs((w2-w)/(w)*100)))
            try:   
                _,e,_ = np.linalg.svd(w2-w)
                pr = np.sum(e.real)**2/np.sum(e.real**2)
                dim.append(pr)
            except:
                dim.append(np.nan)       
    return dif,dim

def calc_err(dat,tar):
    return np.sqrt(np.mean((dat[:,50:]-tar[:,50:])**2))    

datdir = 'results/'
figdir = 'figs/test_fig6_'
teston = 'test_set5'
pert = 'noupstream'
name = 'test'
noisefac = 10
pert2 = 'vmr'
code_order = ['wihl0b','whhl0b','rnn0bto0','wihl0','whhl0','wihl1','whhl1']
code_names = ['Uin','Urec','Uout','Input','PMd rec.','PMd->M1','M1 rec.']

code = {'wihl0':'rnn_l0.weight_ih_l0',
            'whhl0':'rnn_l0.weight_hh_l0',
            'wihl1':'rnn_l1.weight_ih_l0',
            'whhl1':'rnn_l1.weight_hh_l0',
            'wout':'output.weight','bout':'output.bias',
            'wihl0b':'rnn_l0b.weight_ih_l0',
            'whhl0b':'rnn_l0b.weight_hh_l0',
            'rnn0bto0':'rnn0bto0.weight'}


#%%
performance0 = np.zeros((len(pert2),2))*np.nan
whichtype = ['activity1','activity2']
dpsth = np.zeros((len(whichtype)))*np.nan
dcorr = np.zeros((len(whichtype)))*np.nan
datname = datdir+name+'/'
dinit = np.load(datname+'training.npy',allow_pickle=True).item()
# PERFORMANCE
datname = datdir+name+'/null/'
dinit2 = np.load(datname+'testing_'+pert2+'.npy',allow_pickle=True).item()[teston]
performance0[0] = calc_err(dinit2['output'],dinit2['target'])
datname = datdir+name+'/'+pert+'/'
dinit2 = np.load(datname+'testing_'+pert2+'.npy',allow_pickle=True).item()[teston]
performance0[1] = calc_err(dinit2['output'],dinit2['target'])
# ACITIVTY CHANGE
datname = datdir+name+'/'
dinit = np.load(datname+'testing.npy',allow_pickle=True).item()    
for ii,wt in enumerate(whichtype):  
    # psth
    tid0 = dinit[teston]['trial_id']
    act0 = dinit[teston][wt][:,(282-60):(282+60)]
    tid = dinit2['trial_id']
    act = dinit2[wt][:,(282-60):(282+60)]
    psth = []
    psth2 = []
    for jj in range(8):
        psth.append(np.mean(act0[tid0==jj],axis=0))
        psth2.append(np.mean(act[tid==jj],axis=0))
    psth = np.array(psth)
    psth2 = np.array(psth2)
    norm = np.std(psth,axis=(0,1))
    dif = (psth2-psth)/norm[None,None,:]*100
    dpsth[ii] = np.median(abs(dif[:,:,:]))
    # corr
    c1 = np.cov(psth[:,:].reshape(-1,psth.shape[-1]).T)
    c2 = np.cov(psth2[:,:].reshape(-1,psth.shape[-1]).T)
    dcorr[ii] = np.corrcoef(c1.ravel(),c2.ravel())[0,1]

#%% perturbation analysis (add synaptic noise) 
# set up for simulation
from toolbox_pytorch import RNN
import torch

whichtype = ['activity1','activity2']
performance = np.zeros((5))*np.nan
dimsrandom = []
wrandom = []
datname = datdir+name+'/'
dinit = np.load(datname+'testing.npy',allow_pickle=True).item()   
# load data
rundata0 = np.load(datdir+name+'/training.npy',allow_pickle=True).item()
testdata0 = np.load(datdir+name+'/testing.npy',allow_pickle=True).item()
params2 = rundata0['params']
data = np.load(params2['datname']+'.npy',allow_pickle = True).item()
testdata = data['test_set5']
params = data['params']

tsteps = params['tsteps']
input_dim = params['input_dim']
output_dim = params['output_dim']
n1 = params2['n1']
n2 = params2['n2']
dt = params2['dt']
tau = params2['tau']
dtype = torch.FloatTensor
# create model instance
model = RNN(input_dim, output_dim, n1, n2, dt/tau, dtype)
# load original setup
initial_training = torch.load(datdir+name+'/model',map_location=torch.device('cpu'))
model.load_state_dict(initial_training['model_state_dict'])
original = testdata0['test_set5']
weights0 = rundata0['params1']
# load adapted setup   
datname = datdir+name+'/'+pert+'/'
dpert = np.load(datname+'training_'+pert2+'.npy',allow_pickle=True).item()
dpert2 = np.load(datname+'testing_'+pert2+'.npy',allow_pickle=True).item()
perturbed = dpert2['test_set5']
dpert_training = torch.load(datname+'/model_'+pert2,map_location=torch.device('cpu'))['model_state_dict']
    
dif = {}
dimtemp = []
wtemp = []
for k in dpert['params1'].keys():
    if k=='ff_mask' or k=='rnn0bto0_mask':
        continue
    w = dpert['params1'][k]
    w2 = dpert['params2'][k]
    dif.update({k:(w2-w)})

state_dict = model.state_dict()
for j in range(len(code_order)):
    temp0 = weights0[code_order[j]].copy()
    rand_pattern = dif[code_order[j]].copy()
    randompattern = np.random.randn(*rand_pattern.shape)*np.std(rand_pattern)*noisefac
    rand_pattern += randompattern
    temp = temp0 + rand_pattern
    w = temp0
    w2 = temp
    wtemp.append(np.median(abs((w2-w)/(w)*100)))
    
    state_dict[code[code_order[j]]] = torch.FloatTensor(temp)
    try:
        _,e,_ = np.linalg.svd(randompattern)
        pr = np.sum(e.real)**2/np.sum(e.real**2)
    except:
        pr = np.nan
    dimtemp.append(pr)
wrandom.append(wtemp)
dimsrandom.append(dimtemp)       
model.load_state_dict(state_dict, strict=True)
# run simulation
stimtmp = np.zeros((testdata['stimulus'].shape[0],tsteps,input_dim))
stimtmp[:,:,:2] = testdata['stimulus'][:,:,:2]
stimtmp[:,:,2] = testdata['stimulus'][:,:,-1]
stim = torch.Tensor(stimtmp.transpose(1,0,2)).type(dtype)
# model run
testout,testl1,testl2,testl1b = model(stim)
# save it
output = testout.cpu().detach().numpy().transpose(1,0,2)
activity1 = testl1.cpu().detach().numpy().transpose(1,0,2)
activity2 = testl2.cpu().detach().numpy().transpose(1,0,2)   
actdict = {'activity1':activity1,'activity2':activity2}
performance[0] = calc_err(output, perturbed['target'])
dinit2 = np.array([activity1,activity2])
# ACITIVTY CHANGE 
for ii,wt in enumerate(whichtype):  
    # psth
    tid0 = dinit[teston]['trial_id']
    act0 = dinit[teston][wt][:,(282-60):(282+60)]
    tid = dinit[teston]['trial_id']
    act = dinit2[ii][:,(282-60):(282+60)]
    psth = []
    psth2 = []
    for jj in range(8):
        psth.append(np.mean(act0[tid0==jj],axis=0))
        psth2.append(np.mean(act[tid==jj],axis=0))
    psth = np.array(psth)
    psth2 = np.array(psth2)
    norm = np.std(psth,axis=(0,1))
    dif = (psth2-psth)/norm[None,None,:]*100
    performance[1+ii] = np.median(abs(dif[:,:,:]))
    # corr
    c1 = np.cov(psth[:,:].reshape(-1,psth.shape[-1]).T)
    c2 = np.cov(psth2[:,:].reshape(-1,psth.shape[-1]).T)
    performance[3+ii] = np.corrcoef(c1.ravel(),c2.ravel())[0,1]

performance = performance[None,]
dpsth = dpsth[None,]
dcorr = dcorr[None,]
dimsrandom = np.array(dimsrandom)
wrandom = np.array(wrandom)
#%% plot
lab = ['No noise','Noise']
plt.figure(figsize=(1.2,2))
plt.bar(0,np.mean(performance0[:,1],axis=0),
        yerr=np.std(performance0[:,1],axis=0),color='k', width=0.8)
plt.bar(1,np.mean(performance[:,0],axis=0),
        yerr=np.std(performance[:,0],axis=0),facecolor='white',edgecolor='k', width=0.8)
m=np.nanmean(performance0[:,0],axis=0)
plt.plot([-0.4,0.4],[m,m],'--',color='k')
plt.xticks(range(2),lab)
plt.xticks(rotation=60,ha='right')
plt.ylabel('Performance error (cm)')
plt.savefig(figdir+'weightnoise_mse.svg',bbox_inches='tight')

lab = ['PMd','PMd\n(N)','M1','M1\n(N)']
plt.figure(figsize=(1.8,2))
plt.bar(0,np.mean(dpsth[:,0],axis=0),
        yerr=np.std(dpsth[:,0],axis=0),color=col['pmd'])
plt.bar(1,np.mean(performance[:,1],axis=0),
        yerr=np.std(performance[:,1],axis=0),facecolor='white',edgecolor=col['pmd'])
plt.bar(2,np.mean(dpsth[:,1],axis=0),
        yerr=np.std(dpsth[:,1],axis=0),color=col['m1'])
plt.bar(3,np.mean(performance[:,2],axis=0),
        yerr=np.std(performance[:,2],axis=0),facecolor='white',edgecolor=col['m1'])
plt.ylabel('Change in activity (%)')
plt.xticks(range(4),lab)
plt.ylim(0,100)
plt.savefig(figdir+'weightnoise_dpsth.svg',bbox_inches='tight')

plt.figure(figsize=(1.8,2))
plt.bar(0,np.mean(1-dcorr[:,0],axis=0),
        yerr=np.std(1-dcorr[:,0],axis=0),color=col['pmd'])
plt.bar(1,np.mean(1-performance[:,3],axis=0),
        yerr=np.std(1-performance[:,3],axis=0),facecolor='white',edgecolor=col['pmd'])
plt.bar(2,np.mean(1-dcorr[:,1],axis=0),
        yerr=np.std(1-dcorr[:,1],axis=0),color=col['m1'])
plt.bar(3,np.mean(1-performance[:,4],axis=0),
        yerr=np.std(1-performance[:,4],axis=0),facecolor='white',edgecolor=col['m1'])

plt.ylabel('Change in covariance')
plt.xticks(range(4),lab)
plt.ylim(0,1)
plt.savefig(figdir+'weightnoise_dcorr.svg',bbox_inches='tight')

# dim plot
plt.figure(figsize=(1.8,2))
plt.bar(code_names,np.nanmean(dimsrandom,axis=0),
        yerr=np.nanstd(dimsrandom,axis=0))
plt.xticks(rotation=60,ha='right')
plt.ylabel('Dimensionality of\nweight change')
plt.savefig(figdir+'weightnoise_dim.svg',bbox_inches='tight')

plt.figure(figsize=(1.8,2))
plt.bar(code_names,np.nanmean(wrandom,axis=0),
        yerr=np.nanstd(wrandom,axis=0))
plt.xticks(rotation=60,ha='right')
plt.xlim(2.1,6.9)
plt.ylabel('Change in connection\nweights (%)')   
plt.savefig(figdir+'weightnoise_change.svg',bbox_inches='tight')

#%% plot illustration
datname = datdir+name+'/noupstream/'
dinit = np.load(datname+'training_'+pert2+'.npy',allow_pickle=True).item()
w1 = dinit['params1']['whhl1']
w2 = dinit['params2']['whhl1']
dif = w2-w1
maxv = np.max(abs(w1))
_,e,_ = np.linalg.svd(dif)
prd = np.sum(e.real)**2/np.sum(e.real**2) 
_,e,_ = np.linalg.svd(w1)
pr = np.sum(e.real)**2/np.sum(e.real**2) 
noisefac = 10
rand_pattern = dif.copy()
randompattern = np.random.randn(*rand_pattern.shape)*np.std(rand_pattern)*noisefac
rand_pattern += randompattern
w2noise = w1 + rand_pattern
_,e,_ = np.linalg.svd(w2noise-w1)
prn = np.sum(e.real)**2/np.sum(e.real**2) 

plt.figure()
plt.imshow(w2noise-w1,vmin=-maxv/5,vmax=maxv/5,cmap=plt.cm.bwr_r,interpolation='None')
plt.title('Change in connection weights + noise')
plt.text(0,0,'Dim.=%d'%prn)
plt.colorbar(shrink=0.8)
plt.savefig(figdir+'_matrix3.svg',bbox_inches='tight')