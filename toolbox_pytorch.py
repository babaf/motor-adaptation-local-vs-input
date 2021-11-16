import torch
import torch.nn as nn
from collections import OrderedDict #nice to print from

class RNN(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_neurons_l0, n_neurons_l1, alpha, 
                 dtype):
        super(RNN, self).__init__()
        
        self.n_neurons_l0 = n_neurons_l0
        self.n_neurons_l1 = n_neurons_l1
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.alpha = alpha

        self.rnn_l0 = nn.RNN(n_inputs, n_neurons_l0, num_layers=1,
                                nonlinearity='tanh',bias=False) 
        self.rnn_l1 = nn.RNN(n_neurons_l0, n_neurons_l1, num_layers=1,
                                nonlinearity='tanh',bias=False) 
        self.output = nn.Linear(n_neurons_l1, n_outputs)

        self.dtype = dtype
        
        # upstream module
        self.rnn_l0b = nn.RNN(n_inputs, n_neurons_l0, num_layers=1,
                                nonlinearity='tanh',bias=False) 
        self.rnn0bto0 = nn.Linear(n_neurons_l0, n_neurons_l0,bias=False)
        
    def init_hidden(self):
        return ((torch.rand(1,self.batch_size, self.n_neurons_l0)-0.5)*0.2).type(self.dtype), \
                ((torch.rand(1,self.batch_size, self.n_neurons_l1)-0.5)*0.2).type(self.dtype), \
                ((torch.rand(1,self.batch_size, self.n_neurons_l0)-0.5)*0.2).type(self.dtype)

      
    def f_step(self,xin,x1,r1,x2,r2,x1b,r1b):
        # update step for PMd
        x1 = x1 + self.alpha*(-x1 + r1 @ self.rnn_l0.weight_hh_l0.T 
                                  + xin @ self.rnn_l0.weight_ih_l0.T
                                  + r1b @ self.rnn0bto0.weight.T 
                             )
        # update step for UPSTREAM
        x1b = x1b + self.alpha*(-x1b + r1b @ self.rnn_l0b.weight_hh_l0.T 
                                  + xin @ self.rnn_l0b.weight_ih_l0.T
                             )
        # update step for M1
        x2 = x2 + self.alpha*(-x2 + r2 @ self.rnn_l1.weight_hh_l0.T
                                  + r1 @ self.rnn_l1.weight_ih_l0.T
                             )
        r1 = x1.tanh()
        r2 = x2.tanh()
        r1b = x1b.tanh()
        return x1,r1,x2,r2,x1b,r1b
    
    def forward(self, X):
        self.batch_size = X.size(1)
        hidden0,hidden1,hidden0b = self.init_hidden()
        x1 = hidden0
        r1 = x1.tanh()
        x1b = hidden0b
        r1b = x1b.tanh()
        x2 = hidden1
        r2 = x2.tanh()
        hiddenl1 = torch.zeros(X.size(0), X.size(1), self.n_neurons_l0).type(self.dtype)
        hiddenl1b = torch.zeros(X.size(0), X.size(1), self.n_neurons_l0).type(self.dtype)
        hiddenl2 = torch.zeros(X.size(0), X.size(1), self.n_neurons_l1).type(self.dtype)
        for j in range(X.size(0)):
            x1,r1,x2,r2,x1b,r1b = self.f_step(X[j],x1,r1,x2,r2,x1b,r1b)
            hiddenl1[j] = r1
            hiddenl1b[j] = r1b
            hiddenl2[j] = r2
        outv = self.output(hiddenl2)
        return outv, hiddenl1, hiddenl2, hiddenl1b
 
    def load_parameters(self, params):
        state_dict = self.state_dict()
        state_dict['rnn_l0.weight_ih_l0'] = torch.Tensor(params['wihl0']).type(self.dtype)
        state_dict['rnn_l0.weight_hh_l0'] = torch.Tensor(params['whhl0']).type(self.dtype)
        state_dict['rnn_l1.weight_ih_l0'] = torch.Tensor(params['wihl1']).type(self.dtype)
        state_dict['rnn_l1.weight_hh_l0'] = torch.Tensor(params['whhl1']).type(self.dtype)
        state_dict['output.weight'] = torch.Tensor(params['wout']).type(self.dtype)
        state_dict['output.bias'] = torch.Tensor(params['bout']).type(self.dtype)
        state_dict['rnn_l0b.weight_ih_l0'] = torch.Tensor(params['wihl0b']).type(self.dtype)
        state_dict['rnn_l0b.weight_hh_l0'] = torch.Tensor(params['whhl0b']).type(self.dtype)
        state_dict['rnn0bto0.weight'] = torch.Tensor(params['rnn0bto0']).type(self.dtype)
        self.load_state_dict(state_dict, strict=True)
        
    def save_parameters(self):
        wihl0 = self.rnn_l0.weight_ih_l0.cpu().detach().numpy().copy()
        whhl0 = self.rnn_l0.weight_hh_l0.cpu().detach().numpy().copy()
        wihl1 = self.rnn_l1.weight_ih_l0.cpu().detach().numpy().copy()
        whhl1 = self.rnn_l1.weight_hh_l0.cpu().detach().numpy().copy()
        wout = self.output.weight.cpu().detach().numpy().copy()
        bout = self.output.bias.cpu().detach().numpy().copy()
        wihl0b = self.rnn_l0b.weight_ih_l0.cpu().detach().numpy().copy()
        whhl0b = self.rnn_l0b.weight_hh_l0.cpu().detach().numpy().copy()
        rnn0bto0 = self.rnn0bto0.weight.cpu().detach().numpy().copy()
        dic = {'wihl0':wihl0,'whhl0':whhl0,
               'wihl1':wihl1,'whhl1':whhl1,
               'wout':wout,'bout':bout,
               'wihl0b':wihl0b,'whhl0b':whhl0b,'rnn0bto0':rnn0bto0
                }
        return dic
    
# TRAINING FUNCTION ##################### 
def train(modelt, optimizert, criteriont, tt, targett, stimt, 
          alp0=1e-3, bet0=1.9*1e-3, gam0=1e-4, 
          alp1=1e-3, bet1=1.9*1e-3, gam1=1e-4, 
          clipv=0.2):

    lc = []
    plot_fac = 1
    for epoch in range(tt): 
        toprint = OrderedDict()
        train_running_loss = 0.0
        modelt.train()
        # one training step
        optimizert.zero_grad()
        output,rl0,rl1,rl0b = modelt(stimt[epoch])
        # calculate loss
        loss = criteriont(output[50:], targett[epoch,50:])      
        loss_train = loss.mean() 
        toprint['Loss'] = loss_train*plot_fac
        
        # add regularization
        # term 1: parameters
        reg0in = alp0*modelt.rnn_l0.weight_ih_l0.norm(2)
        reg0rec = gam0*modelt.rnn_l0.weight_hh_l0.norm(2)
        reg1in = alp1*modelt.rnn_l1.weight_ih_l0.norm(2)
        reg1rec = gam1*modelt.rnn_l1.weight_hh_l0.norm(2)
        reg1out = alp0*modelt.output.weight.norm(2)
        toprint['R_l0inp'] = reg0in*plot_fac
        toprint['R_l0rec'] = reg0rec*plot_fac
        toprint['R_l1inp'] = reg1in*plot_fac
        toprint['R_l1rec'] = reg1rec*plot_fac
        toprint['R_out'] = reg1out*plot_fac
        # term 2: rates
        reg0act = bet0*rl0.pow(2).mean()
        reg1act = bet1*rl1.pow(2).mean()
        toprint['R_l0rate'] = reg0act*plot_fac
        toprint['R_l1rate'] = reg1act*plot_fac
        loss = loss_train+reg1in+reg1rec+reg0in+reg0rec+reg1out+reg1act+reg0act
        
        reg0bact = bet0*rl0b.pow(2).mean()
        reg0bin = alp0*modelt.rnn_l0b.weight_ih_l0.norm(2)
        reg0brec = gam0*modelt.rnn_l0b.weight_hh_l0.norm(2)
        reg0bout = alp0*modelt.rnn0bto0.weight.norm(2)
        loss += reg0bin+reg0brec+reg0bout+reg0bact
        loss.backward()

        #####################################

        torch.nn.utils.clip_grad_norm_(modelt.parameters(), clipv)
        
        optimizert.step()
            
        train_running_loss = [loss_train.detach().item(),reg0in.detach().item(),
                          reg0rec.detach().item(),reg1in.detach().item(),
                          reg1rec.detach().item(),reg1out.detach().item(),
                          reg0act.detach().item(),reg1act.detach().item(),reg0bact.detach().item()]

        modelt.eval()
        print(('Epoch=%d | '%(epoch)) +" | ".join("%s=%.4f"%(k, v) for k, v in toprint.items()))
        lc.append(train_running_loss)   

    return lc
