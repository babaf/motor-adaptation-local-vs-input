"""
Loads simulation data after VR perturbation experiment and plots
average magnitude and dimensionality of connectivity changes of
the the different modules.


Requirement(s):
    - run run_initial.py
    - run run_perturbation with: subname = 'null' (no adaptation)
    - run run_perturbation with: subname = 'noupstream' (Hlocal)
    - run run_perturbation with: subname = 'extrainput' (Hinput)
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
    
datdir = 'results/'
figdir = 'figs/test_fig5_'
name = 'test'
teston = 'test_set5'
pert = ['extrainput','noupstream']
pert2 = 'vmr'
code_order = ['wihl0b','whhl0b','rnn0bto0','wihl0','whhl0','wihl1','whhl1','wout','bout']
code_names = ['->Upstream','Upstream rec.','Upstream->PMd','->PMd','PMd rec.','PMd->M1','M1','Output','Output bias']

code = {'wihl0':'rnn_l0.weight_ih_l0',
            'whhl0':'rnn_l0.weight_hh_l0',
            'wihl1':'rnn_l1.weight_ih_l0',
            'whhl1':'rnn_l1.weight_hh_l0',
            'wout':'output.weight','bout':'output.bias',
            'wihl0b':'rnn_l0b.weight_ih_l0',
            'whhl0b':'rnn_l0b.weight_hh_l0',
            'rnn0bto0':'rnn0bto0.weight'}

# load data
weights = np.zeros((1,len(pert),len(code_order)))*np.nan
weight_dim = np.zeros((1,len(pert),len(code_order)))*np.nan
  
for i in range(len(pert)):
    datname = datdir+name+'/'
    dinit = np.load(datname+'training.npy',allow_pickle=True).item()
    # WEIGHTS
    datname = datdir+name+'/'+pert[i]+'/'
    dinit = np.load(datname+'training_'+pert2+'.npy',allow_pickle=True).item()
    dif,dim = calc_dw(dinit)
    weights[0,i] = dif
    weight_dim[0,i] = dim

#%% plots
plt.figure(figsize=(2,2))
plt.bar(code_names,np.nanmean(weights[:,0],axis=0),
        yerr=np.nanstd(weights[:,0],axis=0), width=0.8)
plt.xticks(rotation=60,ha='right')
plt.xlim(-0.9,2.9)
plt.ylim(0,2)
plt.ylabel('Change in connection\nweights (%)')    
plt.savefig(figdir+pert[0]+'_weights.svg',bbox_inches='tight')

plt.figure(figsize=(2,2))
plt.bar(code_names,np.nanmean(weight_dim[:,0],axis=0),
        yerr=np.nanstd(weight_dim[:,0],axis=0), width=0.8)
plt.xticks(rotation=60,ha='right')
plt.xlim(-0.9,2.9)
plt.ylabel('Dimensionality of\nweight change')    
plt.yticks([0,5,10,15,20])
plt.savefig(figdir+pert[0]+'_dim.svg',bbox_inches='tight')

plt.figure(figsize=(2,2))
plt.bar(code_names,np.nanmean(weights[:,1],axis=0),
        yerr=np.nanstd(weights[:,1],axis=0), width=0.8)
plt.xticks(rotation=60,ha='right')
plt.xlim(2.1,6.9)
plt.ylim(0,2)
plt.ylabel('Change in connection\nweights (%)')    
plt.savefig(figdir+pert[1]+'_weights.svg',bbox_inches='tight')

plt.figure(figsize=(2,2))
plt.bar(code_names,np.nanmean(weight_dim[:,1],axis=0),
        yerr=np.nanstd(weight_dim[:,1],axis=0), width=0.8)
plt.xticks(rotation=60,ha='right')
plt.xlim(2.1,6.9)
plt.ylabel('Dimensionality of\nweight change')    
plt.yticks([0,5,10,15,20])
plt.savefig(figdir+pert[1]+'_dim.svg',bbox_inches='tight')

#%% plot weight matrices
w1 = dinit['params1']['whhl1']
w2 = dinit['params2']['whhl1']
dif = w2-w1
maxv = np.max(abs(w1))
_,e,_ = np.linalg.svd(dif)
prd = np.sum(e.real)**2/np.sum(e.real**2) 
_,e,_ = np.linalg.svd(w1)
pr = np.sum(e.real)**2/np.sum(e.real**2) 

plt.figure()
plt.imshow(dif,vmin=-maxv/10,vmax=maxv/10,cmap=plt.cm.bwr_r,interpolation='None')
plt.title('Change in connection weights')
plt.text(0,0,'Dim.=%d'%prd)
plt.colorbar(shrink=0.8)
plt.savefig(figdir+'_matrix1.svg',bbox_inches='tight')

plt.figure()
plt.imshow(w1,vmin=-maxv,vmax=maxv,cmap=plt.cm.bwr_r,interpolation='None')
plt.title('Initial connection weights')
plt.text(0,0,'Dim.=%d'%pr)
plt.colorbar(shrink=0.8)
plt.savefig(figdir+'_matrix2.svg',bbox_inches='tight')


   