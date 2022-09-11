from itertools import permutations
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

def Heston_EM(n_paths,n_steps,n_dim, mu, kappa,theta,xi,T,rho = 0.5,s0 = 1,v0 = 0.5,limit=torch.inf,periodic = False,sin_co = 2*math.pi) :
    if 2*kappa*theta <= xi**2 : print('!!Feller cond. not respected!!')
    s_drift = lambda x,t,delta : torch.minimum(x,(limit*torch.ones(1)).expand(x.size()))*delta*mu
    s_diffusion = lambda x,v,delta : torch.minimum(x,limit*torch.ones(1))*torch.sqrt(torch.maximum(v,torch.zeros_like(v)))*delta
    if periodic :
        s_drift = lambda x,t,delta : x*delta*mu*(1 + np.sin(sin_co * t))

    v_drift = lambda v,delta : kappa*(theta - v)*delta
    v_diffusion = lambda v,delta : xi*torch.sqrt(torch.maximum(v,torch.zeros_like(v)))*delta

    dt = T/n_steps
    S = torch.empty(n_paths,n_steps+1,n_dim) # SHAPE (PATHS, TIMESTEPS, DIMENSION)
    V = torch.empty(n_paths,n_steps+1,n_dim)

    if isinstance(s0,(float,int)) :
        S_0 = s0*torch.ones(n_paths,n_dim)
    else : S_0 = s0
    if isinstance(v0,(float,int)) :
        V_0 = v0*torch.ones(n_paths,n_dim)
    else : V_0 = v0


    S[:,0,:] = S_0
    V[:,0,:] = V_0
    # COV = torch.tensor([[1,rho],[rho,1]])
    # MvN = torch.distributions.MultivariateNormal(torch.zeros(2),COV)
    # dW = MvN.sample((n_paths,n_steps,n_dim)) * math.sqrt(dt)
    dW1 = torch.normal(0,1,(n_paths,n_steps,n_dim)) * math.sqrt(dt)
    dW2 = rho*dW1 + math.sqrt(1 - rho**2) *torch.normal(0,1,(n_paths,n_steps,n_dim)) * math.sqrt(dt)
    dW = torch.stack([dW1,dW2],dim=3)



    for j in range(n_steps) : #EULER - MARUYAMA
        t = (j + 1)*dt
        S[:,j+1,:] = S[:,j,:] + s_drift(S[:,j,:],t,dt) + s_diffusion(S[:,j,:],V[:,j,:],dW[:,j,:,0])
        V[:,j+1,:] = V[:,j,:] + v_drift(V[:,j,:],dt) + v_diffusion(V[:,j,:],dW[:,j,:,1])

    out = {
        'stock' : S,
        'vol' : V,
        'dw' : dW
    }

    return out

def Expected_Next_Obs_limitHeston(Observations,Delta,mu,limit = 10) :
    coeff = mu*Delta
    ind1 = torch.nonzero(Observations >= limit,as_tuple=True)
    ind2 = torch.nonzero(Observations < limit,as_tuple=True)
    out = torch.zeros_like(Observations)
    out[ind1] = Observations[ind1] + mu*limit*Delta[ind1]
    tempout = Observations[ind2]*torch.exp(coeff[ind2])
    index1 = torch.nonzero(tempout > limit,as_tuple=True)
    delta1 = (Delta[ind2])[index1] - (torch.log(limit/(Observations[ind2][index1])))/mu #where E_t[X_t] = limit
    tempout[index1] = (limit + mu*limit*delta1)
    out[ind2] = tempout
    return out


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
                K = random.sample(range(1,nsteps-1),N-1) #n-1 observation after 0
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
        TS_M = torch.ones(nsteps+1)
        TS_S = torch.ones(nsteps+1)
        TS_Max = torch.ones(nsteps+1)
        for ts in range(1,nsteps + 1) :
            if ts != nsteps- 1 :
                ind = torch.nonzero(T_Steps == ts,as_tuple=True)
                tsmean = torch.mean(O[ind])
                tsstd  = torch.std(O[ind])
                tsmax  = torch.max(torch.abs(O[ind]))
                TS_M[ts] = tsmean
                TS_S[ts] = tsstd
                TS_Max[ts] = tsmax
        TS_M[0] = TS_M[1]
        TS_S[0] = TS_S[1]
        TS_Max[0] = TS_Max[1]
        self.TS_M = TS_M
        self.TS_S = TS_S
        self.TS_Max = TS_Max

    
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
                "TS_M": self.TS_M,
                "TS_S": self.TS_S,
                "TS_Max": self.TS_Max,
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
