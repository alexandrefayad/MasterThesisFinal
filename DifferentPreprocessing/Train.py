import math
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
import argparse
import utils
import Network
from torch.utils.data import DataLoader
import random
from pathlib import Path
from tqdm import tqdm


parser = argparse.ArgumentParser(description='parameters')

#no need to touch those
parser.add_argument('--npaths',type = int,default=20000,help = 'number of paths')
parser.add_argument('--nsteps',type = int,default=100,help = 'number of time steps to generate the BM')
parser.add_argument('--ndim',type = int,default=1,help = 'number of dimensions')
parser.add_argument('--T',type = float,default=1,help = 'time of maturity')
parser.add_argument('--mu',type = float,default=2,help = 'drift coeff')
parser.add_argument('--sigma',type = float,default=1,help = 'diffustion coeff')
parser.add_argument('--kappa',type = float,default=2,help = 'speed of mean reversing')
parser.add_argument('--theta',type = float,default=1.,help = 'mean value of vol')
parser.add_argument('--xi',type = float,default=1,help = 'variance of volatility')
parser.add_argument('--rho',type = float,default=0.5,help = 'correlation')
parser.add_argument('--s0',type = float,default=1.,help = 'initial stock value')
parser.add_argument('--v0',type = float,default=1.,help = 'initial volatility value')

parser.add_argument('--N',type = int,default=10,help = 'number of observations done')
parser.add_argument('--regular',default = False ,action = 'store_true',help = 'regular observations')
parser.add_argument('--periodic',default = False ,action = 'store_true',help = 'periodic drift')
parser.add_argument('--sin',type = float,default=2,help = 'sin coeff (to be mult by pi)')
parser.add_argument('--NMC',type = int,default=10,help = 'number of runs for monte-carlo approximation of evaluations')
parser.add_argument('--ModelType', type = str, default= 'LEM', help = 'model to use choose between Noisy, Stochastic, ODE')
parser.add_argument('--nhid', type=int, default=32, help='hidden size')
parser.add_argument('--epochs', type=int, default=200, help='max epochs')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
parser.add_argument('--batch', type=int, default=100, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--SDE', type = str, default = 'GBM', help = 'SDE to use choose between GBM, OU')
parser.add_argument('--limit',type = float,default=10,help = 'limit of X_t in the drift')
parser.add_argument('--L', default=False, action='store_true', help='use limit version of Heston')


#seed changes the results quite a lot
parser.add_argument('--seed', type=int, default=909, help='random seed')
parser.add_argument('--version', type=int, default=2, help='model version')

parser.add_argument('--stand',default = False ,action = 'store_true',help = 'standarize data')
parser.add_argument('--norm2',default = False ,action = 'store_true',help = 'normalize 2')
parser.add_argument('--TSstand',default = False ,action = 'store_true',help = 'normalize 2')
parser.add_argument('--TSmax',default = False ,action = 'store_true',help = 'normalize 2')
parser.add_argument('--none',default = False ,action = 'store_true',help = 'no normalization')

args = parser.parse_args()

npaths  = args.npaths
nsteps  = args.nsteps
ndim    = args.ndim
T       = args.T
mu      = args.mu
sigma   = args.sigma
kappa   = args.kappa
theta   = args.theta
xi      = args.xi
rho     = args.rho
s0      = args.s0
v0      = theta

N       = args.N
regular = args.regular
periodic = args.periodic
sin_co  = args.sin*math.pi
NMC     = args.NMC
ModelType    = args.ModelType
nhid    = args.nhid
epochs  = args.epochs
device  = args.device
lr      = args.lr
SDE     = args.SDE
limit   = args.limit
LIM     = args.L

seed    = args.seed
version = args.version
batch   = args.batch
dt      = T/nsteps

stand  = args.stand
norm2  = args.norm2
TSstand = args.TSstand
TSmax  = args.TSmax
none   = args.none

if stand :
    norm_type = 'stand'
elif norm2 :
    norm_type = 'norm2'
elif none :
    norm_type = 'none'
elif TSstand :
    norm_type = 'TSstand'
    lr /= 3
elif TSmax :
    norm_type = 'TSmax'
    lr /= 2
else :
    raise ValueError('No normalization type selected')

P = 'P' if periodic else 'A'
params = f'{norm_type}_{SDE}_{P}_{seed}'    
folder = f'Normalization/{version}'
if SDE == 'HEST' :
    if LIM :
        folder = f'LIM/{version}'
        params = f'{norm_type}_{seed}_{LIM}'
    else :
        folder = f'HEST/{version}'
        params = f'{norm_type}_{P}_{seed}'
    
Path(f'{folder}').mkdir(parents=True, exist_ok=True)
f = open(f'{folder}/Param_{params}.txt', 'a')

f.write(args.__str__()  + '\n')

torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

valpaths = int(npaths * 0.5)
if SDE == 'GBM':
    TrainS = utils.GBM(npaths,nsteps,ndim,mu,sigma,T,s0,periodic,sin_co)
    ValS = utils.GBM(valpaths,nsteps,ndim,mu,sigma,T,s0,periodic,sin_co)
elif SDE == 'OU':
    sigma = 4*sigma
    epochs = int(epochs/1.5) 
    TrainS = utils.Ornstein_Uhlenbeck(npaths,nsteps,ndim,mu,sigma,theta,T,s0,periodic)
    ValS = utils.Ornstein_Uhlenbeck(valpaths,nsteps,ndim,mu,sigma,theta,T,s0,periodic)
elif SDE == 'HEST' :
    if not LIM :
        limit = torch.inf
    TrainO = utils.Heston_EM(npaths,nsteps,ndim,mu,kappa,theta,xi,T,rho,s0,v0,limit)
    TrainS = TrainO['stock']
    ValO = utils.Heston_EM(int(0.3*npaths),nsteps,ndim,mu,kappa,theta,xi,T,rho,s0,v0,limit)
    ValS = ValO['stock']
    

TrainDS = utils.RegularDataset(TrainS,N,dt,regular)
ValDS = utils.RegularDataset(ValS,N,dt,regular)
DL_tr = DataLoader(TrainDS, batch)
DL_val = DataLoader(ValDS, valpaths)
print('DATALOADED')

if version == 3 :
    model = Network.LEMStoch(ndim+1,nhid,ndim,0,0).to(device)
elif version == 5 :
    model = Network.LSTM(ndim+1,nhid,ndim).to(device)
else :
    model = Network.LEMStoch(ndim+1,nhid,ndim,1,1,version).to(device)

for j in DL_tr :
    if norm2 :
        S = j["Max"][0].to(device)
        M = torch.zeros((1)).to(device)
    elif stand :
        M = j['Mean'][0].to(device)
        S = j['Std'][0].to(device)
    elif TSstand :
        M_total = j['TS_M'][0]
        S_total = j['TS_S'][0]
        M_total = torch.cummax(M_total,0)[0]
        S_total = torch.cummax(S_total,0)[0]
    elif TSmax :
        S_total = j['TS_Max'][0]
        M_total = torch.zeros_like(S_total)
        S_total = torch.cummax(S_total,0)[0]
    elif none :
        M = torch.zeros((1)).to(device)
        S = torch.ones((1)).to(device)
    break

objective = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = lr,eps = 1e-8)
Training_loss = torch.zeros((epochs))
Val_loss = torch.zeros((epochs))

best_eval_loss = 1000


for epoch in tqdm(range(epochs)) :
    model.train()
    loss = 0
    for j in DL_tr :
        optimizer.zero_grad()
        Observ = j["Observ"].to(device) 
        if TSstand or TSmax :
            TS = j['T_Steps']
            M = torch.gather(M_total.expand(Observ.size(0),nsteps+1),1,TS).unsqueeze(-1).to(device)
            S = torch.gather(S_total.expand(Observ.size(0),nsteps+1),1,TS).unsqueeze(-1).to(device)

        Observ = (Observ - M)/S
        InObserv = Observ[:,:-1,:] #witout the last observation to be used for input
        OutObserv = Observ[:,1:,:] #without the first observation to be used for output
        Delta = j['Delta'].to(device)
        out = model(InObserv.permute(1,0,2),Delta.permute(1,0,2)).permute(1,0,2)
        loss = objective(OutObserv,out)
        Training_loss[epoch] += loss.item()
        loss.backward()
        optimizer.step()



    Training_loss[epoch] /= DL_tr.__len__()

    model.eval()

    with torch.no_grad() :
        for val in DL_val :
            Delta = val['Delta'].to(device)
            Observ = val["Observ"].to(device)
            InObserv = Observ[:,:-1,:]
            if TSstand or TSmax :
                TS = val['T_Steps']
                M2 = torch.gather(M_total.expand(Observ.size(0),nsteps+1),1,TS).unsqueeze(-1).to(device)
                S2 = torch.gather(S_total.expand(Observ.size(0),nsteps+1),1,TS).unsqueeze(-1).to(device)
            else : 
                M2 = M.expand((Observ.size(0),N+1,1)).to(device)
                S2 = S.expand((Observ.size(0),N+1,1)).to(device)


            out = torch.zeros_like(InObserv)
            for K in range(NMC) :   #NMC is the number of Monte Carlo simulations
                out += model(((InObserv - M2[:,:-1,:])/S2[:,:-1,:]).permute(1,0,2),Delta.permute(1,0,2)).permute(1,0,2)
            out = out/NMC
            out = out*S2[:,1:,:] + M2[:,1:,:]
            if SDE == 'GBM' or SDE == 'HEST' :
                expected = utils.Expected_Next_Obs_GBM(InObserv,Delta,mu,periodic,sin_co)
            elif SDE == 'OU':
                expected = utils.Expected_Next_Obs_OU(InObserv,Delta,mu,theta,periodic)
            Val_loss[epoch] += mse_loss(out,expected).item()
    Val_loss[epoch] /= DL_val.__len__()

    log = 'Epoch: {:03d}, Train loss : {:.6f}, Val loss: {:.6f} \n'
    f.write(log.format(epoch+1, Training_loss[epoch],Val_loss[epoch]))
    if (Val_loss[epoch] < best_eval_loss) :
        utils.SaveModelMS(model,f'{folder}/beststate_{params}.pt',M,S)
        best_eval_loss = Val_loss[epoch]

    if epoch == (epochs//2 -1):
    #divide learning rate by 5 after half the steps
        for param_group in optimizer.param_groups :
            param_group['lr'] /= 5

        
    
torch.save(Training_loss,f'{folder}/training_loss_{params}.pt')
torch.save(Val_loss,f'{folder}/val_loss_{params}.pt')
if TSstand or TSmax :
    M = M_total
    S = S_total
utils.SaveModelMS(model,f'{folder}/finalstate_{params}.pt',M,S)
print(f'DONE, last val loss : {Val_loss[-1]:.7f}')
f.close()