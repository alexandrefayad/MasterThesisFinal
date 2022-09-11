import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import random


def GBM(n_paths,n_steps,n_dim, mu,sigma, T,x0, periodic = False,sin_co = 2*math.pi) :
    if periodic :
        drift = lambda x,t,delta : x*delta*mu*(1 + np.sin(sin_co * t))
    else : 
        drift = lambda x,t,delta : x*delta*mu
    diffusion = lambda x,delta : x*delta*sigma
    dt = T/n_steps
    Paths = torch.empty(n_paths,n_steps+1,n_dim) # SHAPE (PATHS, TIMESTEPS, DIMENSION)

    if isinstance(x0,(float,int)) :
        X_0 = x0*torch.ones(n_paths,n_dim)
    else : X_0 = x0
    
    Paths[:,0,:] = X_0
    dW = torch.normal(0,1,(n_paths,n_steps,n_dim)) * math.sqrt(dt)

    for j in range(n_steps) : #EULER - MARUYAMA
        t = (j + 1)*dt
        Paths[:,j+1,:] = (Paths[:,j,:] + drift(Paths[:,j,:],t,dt) + diffusion(Paths[:,j,:],dW[:,j,:]))

    return Paths

def Expected_Next_Obs_GBM(Observations,Delta,mu,periodic = False, sin_co = 2*math.pi) :
    # To use for the second usage of RNN (i.e: when we only let the observations as input)
    if periodic :
        Times = torch.cat([torch.zeros_like(Delta[:,0,:].unsqueeze(1)),Delta.cumsum(1)],1) # times + 0 for the first obs. time
        coeff = mu*(Delta - 1/(sin_co + 1e-10) *(torch.cos(sin_co * Times[:,1:,:]) - torch.cos(sin_co * Times[:,:-1,:])))
    else :
        coeff = mu*Delta    
        
    out = Observations*(torch.exp(coeff))
    return out

def Ornstein_Uhlenbeck(n_paths,n_steps,n_dim, mu, sigma,theta,T,s0,periodic = False) :
    if periodic :
        drift = lambda x,t,delta : theta*(mu*t - x)* delta
    else :
        drift = lambda x,t,delta : theta*(mu - x)* delta
    diffusion = lambda delta : sigma*delta

    dt = T/n_steps
    S = torch.empty(n_paths,n_steps+1,n_dim) # SHAPE (PATHS, TIMESTEPS, DIMENSION)

    if isinstance(s0,(float,int)) :
        S_0 = s0*torch.ones(n_paths,n_dim)
    else : S_0 = s0

    S[:,0,:] = S_0
    dW = torch.normal(0,1,(n_paths,n_steps,n_dim)) * math.sqrt(dt)


    for j in range(n_steps) : #EULER - MARUYAMA
        t = j*dt
        S[:,j+1,:] = S[:,j,:] + drift(S[:,j,:],t,dt) + diffusion(dW[:,j,:])

    return S

def Expected_Next_Obs_OU(Observations,Delta,mu,theta,periodic = False) :
# To use for the second usage of RNN (i.e: when we only let the observations in)
    if periodic :
        Times = torch.cat([torch.zeros_like(Delta[:,0,:].unsqueeze(1)),Delta.cumsum(1)],1) # times + 0 for the first obs. time
        t = Times[:,1:,:]
        s = Times[:,:-1,:]
        out = mu*(t) - mu/theta + (-mu*s + Observations + mu/theta)* torch.exp(-theta*Delta)
    else :
        coeff = -theta*Delta 
        out = Observations*(torch.exp(coeff)) + mu*(1 - torch.exp(coeff))
    return out


def plot_path_expectaion(Path,Delta,expected,output,T = 1,dt = 0.01) :
    fig,ax = plt.subplots()
    ax.plot(np.arange(0,T+dt-1e-10,dt),Path,label = 'True Path')
    ax.plot(torch.cumsum(Delta.to('cpu'),0),expected.to('cpu'),'o',label = 'Expected',color = 'g',)
    ax.plot(torch.cumsum(Delta.to('cpu'),0),output.to('cpu'),'x',label = 'predicted',color = 'orange',markeredgewidth=1.8)
    plt.legend()

class RegularDataset(Dataset) :
    """
    Here number of obs are constant
    """
    def __init__(self,Paths,N,dt,regular) :
        self.Paths = Paths
        self.dt = dt
        nsteps = Paths.shape[1] -1
        npaths = Paths.shape[0]
        T_Steps = torch.empty((npaths,N+1),dtype=int)

        if regular :
            ts = torch.arange(0,nsteps+1,math.floor((nsteps)/N))
            T_Steps = ts.expand((Paths.shape[0],ts.shape[0]))
        else :
            for i in range(Paths.shape[0]) :
                K = random.sample(range(1,nsteps),N-1) #n-1 observation after 0
                K.append(0) #for everyone
                K.append(nsteps)
                K.sort()
                ts = torch.tensor(K,dtype=int)
                T_Steps[i,:] = ts

        Delta = torch.unsqueeze((T_Steps[:,1:] - T_Steps[:,:-1])*dt,-1)
        self.Dt = Delta
        O = Get_Observed(Paths,T_Steps)
        self.Observ = O
        self.T_Steps = T_Steps
        self.mean = torch.mean(O)
        self.std  = torch.std(O)
        self.max  = torch.max(torch.abs(O))

    
    def __len__(self):
        return self.Paths.shape[0]

    def __getitem__(self,idx):
        # stock_path dimension: [BATCH_SIZE, NUMBER OF STEPS,DIMESNION]
        return {"Paths": self.Paths[idx,:,:], 
                "Delta": self.Dt[idx,:,:],
                "Observ": self.Observ[idx,:,:],
                "T_Steps": self.T_Steps[idx,:],
                "Mean": self.mean,
                "Std": self.std,
                "Max": self.max,
                }

def Get_Observed(Paths,TS) :
    out = torch.empty(Paths.size(0),TS.size(1),Paths.size(2))
    for t in range(TS.size(1)) :
        obs = Paths[torch.arange(Paths.size(0)), TS[:,t]]
        out[:,t,:] =obs

    return out

def SaveModelMS(model,path,M,S) :
    dict = {'model':model.state_dict(),'M':M,'S':S}
    torch.save(dict,path)

# def variable_time_collate_fn(batch,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) :
#     dt = batch[0]['dt']
#     comb_ts = torch.unique(torch.cat([batch[i]['Time_steps'] for i in range(len(batch))],0))
#     n_paths = len(batch)
#     n_steps,n_dim = batch[0]['Paths'].size()
#     Mask = torch.zeros((n_paths,n_steps),dtype=bool)
#     Paths= torch.zeros((n_paths,n_steps,n_dim),dtype= torch.float16)
#     delta     = (comb_ts[1:] - comb_ts[:-1])*dt

#     # Time_steps = []
#     Expectations = []
#     for i,b in enumerate(batch) :
#         Mask[i,...]  = b['Mask']
#         Paths[i,...] = b['Paths']
        
        

#     # Observation_indices = torch.nonzero(Mask)
#     # combined_ts = torch.unique(Observation_indices[:,1])
#     # if (torch.nonzero(combined_ts - combined_ts1).nelement() != 0) : print('error')
#     New_Paths = Paths[:,combined_ts]
#     New_Mask  = Mask[:,combined_ts]
#     data_dict = {
#         'Paths' : New_Paths.to(device),
#         'Mask'  : New_Mask.to(device),
#         'Delta' : delta.to(device),
#         # 'Time_steps' : Time_steps,
#         'Expectations' : Expectations,
#         }
#     return data_dict
