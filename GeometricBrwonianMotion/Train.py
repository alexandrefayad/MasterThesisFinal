from ast import In
import math
import os,sys,inspect
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
parser.add_argument('--theta', type = float, default=1, help='theta')
parser.add_argument('--s0',type = float,default=1.,help = 'initial stock value')
parser.add_argument('--N',type = int,default=10,help = 'number of observations done')
parser.add_argument('--regular',default = False ,action = 'store_true',help = 'regular observations')
parser.add_argument('--periodic',default = False ,action = 'store_true',help = 'periodic drift')
parser.add_argument('--sin',type = float,default=2,help = 'sin coeff (to be mult by pi)')
parser.add_argument('--NMC',type = int,default=10,help = 'number of runs for monte-carlo approximation of evaluations')
parser.add_argument('--ModelType', type = str, default= 'LEM', help = 'model to use choose between Noisy, Stochastic, ODE')
parser.add_argument('--nhid', type=int, default=32, help='hidden size')
parser.add_argument('--epochs', type=int, default=150, help='max epochs')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
parser.add_argument('--batch', type=int, default=100, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--SDE', type = str, default = 'GBM', help = 'SDE to use choose between GBM, OU')
parser.add_argument('--long',default = False ,action = 'store_true',help = 'periodic drift')

parser.add_argument('--SetSeed', type=int, default=909, help='random seed')
parser.add_argument('--ModelSeed', type=int, default=909, help='random seed')
parser.add_argument('--version', type=int, default=2, help='model version')

args = parser.parse_args()

npaths  = args.npaths
nsteps  = args.nsteps
ndim    = args.ndim
T       = args.T
mu      = args.mu
sigma   = args.sigma
theta   = args.theta
s0      = args.s0
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
SetSeed = args.SetSeed
ModelSeed = args.ModelSeed
version = args.version
batch   = args.batch
dt      = T/nsteps
long    = args.long
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
tempstring = currentdir.replace(f'{parentdir}/','')
folder = f'{tempstring}/{version}'
P = 'P' if periodic else 'A'
params = f'{P}_{ModelSeed}_{sigma}_{SetSeed}'
if long : 
    folder = f'{tempstring}/long/{version}'
    # lr = 1e-3
Path(f'{folder}').mkdir(parents=True, exist_ok=True)
f = open(f'{folder}/Param_{params}.txt', 'a')

f.write(args.__str__()  + '\n')

torch.manual_seed(SetSeed)
random.seed(SetSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

valpaths = int(npaths * 0.5)
if SDE == 'GBM':
    TrainS = utils.GBM(npaths,nsteps,ndim,mu,sigma,T,s0,periodic,sin_co)
    ValS = utils.GBM(valpaths,nsteps,ndim,mu,sigma,T,s0,periodic,sin_co)
elif SDE == 'OU':
    sigma = 4*sigma
    epochs = int(epochs/1.5) 
    npaths = npaths//2
    TrainS = utils.Ornstein_Uhlenbeck(npaths,nsteps,ndim,mu,sigma,theta,T,s0,periodic)
    ValS = utils.Ornstein_Uhlenbeck(valpaths,nsteps,ndim,mu,sigma,theta,T,s0,periodic)


TrainDS = utils.RegularDataset(TrainS,N,dt,regular)
ValDS = utils.RegularDataset(ValS,N,dt,regular)
DL_tr = DataLoader(TrainDS, batch)
DL_val = DataLoader(ValDS, valpaths)
print('DATALOADED')

torch.manual_seed(ModelSeed)
random.seed(ModelSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if version == 3 :
    model = Network.LEMStoch(ndim+1,nhid,ndim,0,0).to(device)
elif version == 5 :
    model = Network.LSTM(ndim+1,nhid,ndim).to(device)
elif version == 6 :
    model = Network.LEMODE(ndim,nhid,ndim).to(device)
elif version == 7 :
    model = Network.LEMODE2(ndim,nhid,ndim).to(device)
else :
    model = Network.LEMStoch(ndim+1,nhid,ndim,1,1,version).to(device)

for j in DL_tr :
    S = j["Max"][0].item()
    M = 0
    break

objective = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = lr,eps = 1e-8)
Training_loss = torch.zeros((epochs))
Val_loss = torch.zeros((epochs))

best_eval_loss = 1000
best_early_loss = 0
stopping_possible = False
patience  = 0


for epoch in tqdm(range(epochs)) :
    model.train()
    loss = 0
    for j in DL_tr :
        optimizer.zero_grad()
        Observ = j["Observ"].to(device) 
        Observ = Observ/S
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

            out = torch.zeros_like(InObserv)
            
            for K in range(NMC) :   #NMC is the number of Monte Carlo simulations
                out += model((InObserv/S).permute(1,0,2),Delta.permute(1,0,2)).permute(1,0,2)
            out = out/NMC
            out = out*S
            if SDE == 'GBM':
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

    if (Val_loss[epoch] > best_early_loss) and stopping_possible:
        patience += 1
        best_early_loss = Val_loss[epoch]
        if patience == 5:
            utils.SaveModelMS(model,f'{folder}/earlystate_{params}.pt',M,S)
            stopping_possible = False
    else :
        patience = 0
        best_early_loss = Val_loss[epoch]

    if epoch == (50):
    #divide learning rate by 5 after half the steps
        for param_group in optimizer.param_groups :
            param_group['lr'] /= 10
        stopping_possible = True        
    
torch.save(Training_loss,f'{folder}/training_loss_{params}.pt')
torch.save(Val_loss,f'{folder}/val_loss_{params}.pt')
utils.SaveModelMS(model,f'{folder}/finalstate_{params}.pt',M,S)
print(f'DONE, last val loss : {Val_loss[-1]:.7f}, lr = {lr}')
f.close()

# g = open(f'{folder}/lr_{SetSeed}_{sigma}.txt', 'a')
# g.write(f'lr = {lr}, val loss = {Val_loss[-1]}, early loss = {best_early_loss} \n')
# g.close()