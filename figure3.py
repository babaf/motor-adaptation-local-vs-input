"""
Loads simulation data after initial training and plots activity traces.

Requirement(s):
    - run run_initial.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (2,2)
mpl.rcParams['figure.dpi'] = 96
from mpl_toolkits.mplot3d import Axes3D
from pyaldata import *
import os

cmap = [plt.cm.magma(i) for i in np.linspace(0.1,0.9,8)] # color definition

def calc_psth(dat,tid0):
    act0 = dat[:,(282-60):(282+60)] # time window (282 corresponds to GO signal)
    psth = []
    for jj in range(8):
        psth.append(np.nanmean(act0[tid0==jj],axis=0))
    psth = np.array(psth)
    return psth

# set directories
datdir = 'results/'
figdir = 'figs/test_fig3_'
name = 'test' # name of simulation to plot
teston = 'test_set5' # which data to plot

#%% load data
dinit = np.load(datdir + name +'/testing.npy',allow_pickle=True).item()[teston]

#%% initial run
out = dinit['output']
pmd = dinit['activity1']
m1 = dinit['activity2']
tid = dinit['trial_id']
n_neurons = pmd.shape[-1]

out_psth = calc_psth(out,tid)
pmd_psth = calc_psth(pmd,tid)
m1_psth = calc_psth(m1,tid)

covm = np.cov(m1_psth.reshape(-1,m1.shape[-1]).T)
covp = np.cov(pmd_psth.reshape(-1,pmd.shape[-1]).T)

evm,evecm = np.linalg.eig(covm)
evp,evecp = np.linalg.eig(covp)

time = np.linspace(-0.6,0.6,120) # time window
time2 = np.linspace(0,1,100) # time window

#%% plot trajectories
plt.figure()
for j in range(out.shape[0]):
    plt.plot(out[j,(282):(282+100),0],out[j,(282):(282+100),1],
             color=cmap[(5+tid[j])%8])
plt.savefig(figdir + 'example_output.svg',bbox_inches='tight')

plt.figure(figsize=(2,1))
plt.plot(time2, out[j,(282):(282+100),0],color=cmap[tid[j]],label='x')
plt.plot(time2, out[j,(282):(282+100),1],'--',color=cmap[tid[j]],label='y')
plt.legend()
plt.xlabel('Time rel. to GO (s)')
plt.ylabel('Position (cm)')
plt.savefig(figdir + 'example_output_time.svg',bbox_inches='tight')

#%% plot psth 
nid = 0
plt.figure()
for j in range(8):
    plt.plot(time,pmd_psth[j,:,nid],color=cmap[j])
plt.xlabel('Time rel. to GO (s)')
plt.ylabel('Unit activity')
plt.title('PMd')
plt.savefig(figdir + 'example_pmd_psth.svg',bbox_inches='tight')

nid = 5
plt.figure()
for j in range(8):
    plt.plot(time,m1_psth[j,:,nid],color=cmap[j])
plt.xlabel('Time rel. to GO (s)')
plt.ylabel('Unit activity')
plt.title('M1')
plt.savefig(figdir + 'example_m1_psth.svg',bbox_inches='tight')

#%% plot pca
xi = pmd_psth @ evecp.real
plt.figure()
ax = plt.gcf().add_subplot(111, projection='3d')
for j in range(8):
    ax.plot(xi[j,:,0],xi[j,:,1],xi[j,:,2],color=cmap[j])
    ax.scatter(xi[j,0,0],xi[j,0,1],xi[j,0,2],facecolor='k',edgecolor='k',s=5)
    ax.scatter(xi[j,60,0],xi[j,60,1],xi[j,60,2],facecolor=cmap[j],marker='+')
    ax.scatter(xi[j,-1,0],xi[j,-1,1],xi[j,-1,2],facecolor=cmap[j])
ax.view_init(elev=20, azim=30)
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.title('PMd')
plt.savefig(figdir + 'example_pmd_pca.svg',bbox_inches='tight')

xi = m1_psth @ evecm.real
plt.figure()
ax = plt.gcf().add_subplot(111, projection='3d')
for j in range(8):
    ax.plot(xi[j,:,0],xi[j,:,1],xi[j,:,2],color=cmap[j])
    ax.scatter(xi[j,0,0],xi[j,0,1],xi[j,0,2],facecolor='k',edgecolor='k',s=5)
    ax.scatter(xi[j,60,0],xi[j,60,1],xi[j,60,2],facecolor=cmap[j],marker='+')
    ax.scatter(xi[j,-1,0],xi[j,-1,1],xi[j,-1,2],facecolor=cmap[j])
ax.view_init(elev=20, azim=30-180)
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.title('M1')
plt.savefig(figdir + 'example_m1_pca.svg',bbox_inches='tight')
