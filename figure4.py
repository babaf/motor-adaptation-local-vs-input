"""
Load perturbation experiment and plot it.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (2,2)
mpl.rcParams['figure.dpi'] = 96

col = {'VR':'#FF6B6B','pmd':'#aade87','m1':'#008000'}
cmap = [plt.cm.magma(i) for i in np.linspace(0.1,0.9,8)] 

def color(vp,c):
    for pc in vp['bodies']:
        pc.set_facecolor(c)
        pc.set_edgecolor(c)
    for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
        try:
            temp = vp[partname]
            temp.set_edgecolor(c)
        except:
            pass 
        
def plot_dpsth(a,c=None,savname=None):
    plt.figure(figsize=(1.5,2))
    violin_parts = plt.violinplot(a[:,0],positions=np.array([0]),widths=0.8,showmeans=True)
    color(violin_parts,col['pmd'])
    violin_parts = plt.violinplot(a[:,1],positions=np.array([1]),widths=0.8,showmeans=True)
    color(violin_parts,col['m1'])
    plt.scatter((np.random.rand(a.shape[0])-0.5)*0.4,a[:,0],edgecolor=col['pmd'],s=20,
                marker='o',facecolor='None',alpha=0.6)
    plt.scatter((np.random.rand(a.shape[0])-0.5)*0.4+1,a[:,1],edgecolor=col['m1'],s=20,
                marker='o',facecolor='None',alpha=0.6,label='Adaptation')
    plt.xticks(range(2),['PMd','M1'])
    plt.xlim(-0.5,1.5)
    plt.ylabel('Change in activity (%)')
    plt.ylim(0,100)
    if c is not None: # plot control
        cm = np.mean(c,axis=0)
        plt.plot([-0.2,0.2],[cm[0],cm[0]],'k')
        points = c[:,0]
        plt.scatter(np.random.randn(points.size)*0.05,points,c='k',s=5,marker='.',zorder=50)
        plt.plot([0.8,1.2],[cm[1],cm[1]],'k')
        points = c[:,1]
        plt.scatter(np.random.randn(points.size)*0.05+1,points,c='k',s=5,marker='.',zorder=50)
    if savname is not None:
        plt.savefig(figdir+savname+'.svg',bbox_inches='tight')

def plot_dcorr(a,c=None,savname=None):
    plt.figure(figsize=(1.5,2))
    violin_parts = plt.violinplot(a[:,0],positions=np.array([0]),widths=0.8,showmeans=True)
    color(violin_parts,col['pmd'])
    violin_parts = plt.violinplot(a[:,1],positions=np.array([1]),widths=0.8,showmeans=True)
    color(violin_parts,col['m1'])
    plt.scatter((np.random.rand(a.shape[0])-0.5)*0.4,a[:,0],edgecolor=col['pmd'],s=20,
                marker='o',facecolor='None',alpha=0.6)
    plt.scatter((np.random.rand(a.shape[0])-0.5)*0.4+1,a[:,1],edgecolor=col['m1'],s=20,
                marker='o',facecolor='None',alpha=0.6,label='Adaptation')
    plt.xticks(range(2),['PMd','M1'])
    plt.xlim(-0.5,1.5)
    plt.ylabel('Change in covariance')
    plt.ylim(0,1)
    if c is not None: # plot control
        cm = np.mean(c,axis=0)
        plt.plot([-0.2,0.2],[cm[0],cm[0]],'k')
        points = c[:,0]
        plt.scatter(np.random.randn(points.size)*0.05,points,c='k',s=5,marker='.',zorder=50)
        plt.plot([0.8,1.2],[cm[1],cm[1]],'k')
        points = c[:,1]
        plt.scatter(np.random.randn(points.size)*0.05+1,points,c='k',s=5,marker='.',zorder=50)
    if savname is not None:
        plt.savefig(figdir+savname+'.svg',bbox_inches='tight')
        
def calc_err(dat,targ):
    return np.mean((dat[:,50:]-targ[:,50:])**2)

    
# set directories
datdir = 'results/'
name = 'test'
teston = 'test_set5'
plastA = ['extrainput','noupstream'] 
pert = 'vmr'
whichtype = ['activity1','activity2']

figdir = 'figs/test_fig4_'
    
for run in range(len(plastA)):
    plast = plastA[run]
    performance = np.zeros((3))*np.nan
    dpsth = np.zeros((len(whichtype)))*np.nan
    dcorr = np.zeros((len(whichtype)))*np.nan
    # PERFORMANCE
    # Initial
    datname0 = datdir+name+'/'
    dinit0 = np.load(datname0+'testing.npy',allow_pickle=True).item()[teston]
    performance[0] = calc_err(dinit0['output'],dinit0['target'])
    # VR perturbed
    datname = datname0+'/null/'
    dinit = np.load(datname+'testing_'+pert+'.npy',allow_pickle=True).item()[teston]
    performance[1] = calc_err(dinit['output'],dinit['target'])
    # VR retrained
    datname = datname0+plast+'/'
    dinit = np.load(datname+'testing_'+pert+'.npy',allow_pickle=True).item()[teston]
    performance[2] = calc_err(dinit['output'],dinit['target'])

    # ACITIVTY CHANGE
    for i,wt in enumerate(whichtype):  
        # psth
        tid0 = dinit0['trial_id']
        act0 = dinit0[wt][:,(282-60):(282+60)]
        tid = dinit['trial_id']
        act = dinit[wt][:,(282-60):(282+60)]
        psth = []
        psth2 = []
        for jj in range(8):
            psth.append(np.mean(act0[tid0==jj],axis=0))
            psth2.append(np.mean(act[tid==jj],axis=0))
        psth = np.array(psth)
        psth2 = np.array(psth2)

        norm = np.std(psth,axis=(0,1))
        dif = (psth2-psth)/norm[None,None,:]*100
        dpsth[i] = np.median(abs(dif[:,:,:]))
        # corr
        c1 = np.cov(psth[:,:].reshape(-1,psth.shape[-1]).T)
        c2 = np.cov(psth2[:,:].reshape(-1,psth.shape[-1]).T)
        dcorr[i] = 1-np.corrcoef(c1.ravel(),c2.ravel())[0,1]

    # plot trajectories
    outBL = dinit0['output']
    tid = dinit0['trial_id']
    outlAD = dinit['output']
    
    plt.figure()
    for j in range(50):
        plt.plot(outBL[j,(282):(282+100),0],outBL[j,(282):(282+100),1],
                 color=cmap[(5+tid[j])%8])
    plt.title('Baseline')
    plt.savefig(figdir + plast+'_traj_BL.svg',bbox_inches='tight')
    
    plt.figure()
    for j in range(50):
        plt.plot(outlAD[j,(282):(282+100),0],outlAD[j,(282):(282+100),1],
                 color=cmap[(5+tid[j])%8])
    plt.title('Adapted')
    plt.savefig(figdir + plast + '_traj_lAD.svg',bbox_inches='tight')

    # plot: activity change   
    plot_dpsth(dpsth[None,],savname=plast+'_dpsth')
    
    # plot: covariance change
    plot_dcorr(dcorr[None,],savname=plast+'_dcorr')
        
