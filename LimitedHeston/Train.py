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
import os,sys,inspect

parser = argparse.ArgumentParser(description='parameters')

#no need to touch those
parser.add_argument('--npaths',type = int,default=7000,help = 'number of paths')
parser.add_argument('--nsteps',type = int,default=300,help = 'number of time steps to generate the BM')
parser.add_argument('--ndim',type = int,default=1,help = 'number of dimensions')
parser.add_argument('--T',type = float,default=1,help = 'time of maturity')
parser.add_argument('--mu',type = float,default=2,help = 'drift coeff')
parser.add_argument('--sigma',type = float,default=1,help = 'diffustion coeff')
parser.add_argument('--kappa',type = float,default=1,help = 'speed of mean reversing')
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
parser.add_argument('--limit',type = float,default=10,help = 'limit of X_t in the drift')
parser.add_argument('--L', default=False, action='store_true', help='use limit version of Heston')

parser.add_argument('--SetSeed', type=int, default=909, help='random seed')
parser.add_argument('--ModelSeed', type=int, default=909, help='random seed')
parser.add_argument('--version', type=int, default=2, help='model version')
#FOR VERSIONS 0,1,2 CORRESPOND TO V_0 V_1 V_2, 3 CORRESPONDS TO LEM, 4 TO V_3 AND 5 TO LSTM

args = parser.parse_args()

npaths  = args.npaths
nsteps  = args.nsteps
ndim    = args.ndim
T       = args.T
mu      = args.mu
sigma   = args.sigma
kappa   = args.kappa
theta   = sigma**2
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
limit   = args.limit
LIM     = args.L

SetSeed = args.SetSeed
ModelSeed = args.ModelSeed
version = args.version
batch   = args.batch
dt      = T/nsteps

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
tempstring = currentdir.replace(f'{parentdir}/','')

P = 'P' if periodic else 'A'
if LIM :
    folder = f'{tempstring}/LIM/{version}'
    params = f'{ModelSeed}_{limit}'
else :
    folder = f'{tempstring}/HEST/{version}'
    params = f'{P}_{ModelSeed}_{SetSeed}_{sigma}'
    
Path(f'{folder}').mkdir(parents=True, exist_ok=True)
f = open(f'{folder}/Param_{params}.txt', 'a')

f.write(args.__str__()  + '\n')

valpaths = int(npaths * 0.5)
if not LIM :
    limit = 1e7

torch.manual_seed(SetSeed)
random.seed(SetSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TrainO = utils.Heston_EM(npaths,nsteps,ndim,mu,kappa,theta,xi,T,rho,s0,v0,limit)
TrainS = TrainO['stock']
ValO = utils.Heston_EM(valpaths,nsteps,ndim,mu,kappa,theta,xi,T,rho,s0,v0,limit)
ValS = ValO['stock']
    

TrainDS = utils.RegularDataset(TrainS,N,dt,regular)
ValDS = utils.RegularDataset(ValS,N,dt,regular)
DL_tr = DataLoader(TrainDS, batch)
DL_val = DataLoader(ValDS, valpaths)

#get expected value for validation
for val in DL_val :
    Delta = val['Delta']
    Observ = val["Observ"]
    InObserv = Observ[:,:-1,:]
    TS = val["T_Steps"]
    Val_Observ = utils.Get_Observed(ValO['vol'],TS)
    InVal_Observ = Val_Observ[:,:-1,:]
    expected = utils.Expected_Next_Obs_limitHeston(InObserv,InVal_Observ,Delta,mu,ndim,kappa,theta,xi,rho,limit = limit).to(device)

print('DATALOADED')

torch.manual_seed(ModelSeed)
random.seed(ModelSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if version == 3 :
    model = Network.LEMStoch(ndim+1,nhid,ndim,0,0).to(device)
elif version == 5 :
    model = Network.LSTM(ndim+1,nhid,ndim).to(device)
else :
    model = Network.LEMStoch(ndim+1,nhid,ndim,1,1,version).to(device)

for j in DL_tr :
    S = j["Max"][0].to(device)
    M = torch.zeros((1)).to(device)

objective = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = lr,eps = 1e-8)
Training_loss = torch.zeros((epochs))
Val_loss = torch.zeros((epochs))

best_eval_loss = 1000
best_early_loss = 'No early stopping'
prev_loss = 0
stopping_possible = False
patience  = 0


for epoch in tqdm(range(epochs)) :
    model.train()
    loss = 0
    for j in DL_tr :
        optimizer.zero_grad()
        Observ = j["Observ"].to(device) 

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

            out = torch.zeros_like(InObserv)
            for K in range(NMC) :   #NMC is the number of Monte Carlo simulations
                out += model(((InObserv - M)/S).permute(1,0,2),Delta.permute(1,0,2)).permute(1,0,2)
            out = out/NMC
            out = out*S + M
            Val_loss[epoch] += mse_loss(out,expected).item()
    Val_loss[epoch] /= DL_val.__len__()

    log = 'Epoch: {:03d}, Train loss : {:.6f}, Val loss: {:.6f} \n'
    f.write(log.format(epoch+1, Training_loss[epoch],Val_loss[epoch]))
    if (Val_loss[epoch] < best_eval_loss) :
        utils.SaveModelMS(model,f'{folder}/beststate_{params}.pt',M,S)
        best_eval_loss = Val_loss[epoch]

    #early stopping
    if (Val_loss[epoch] > prev_loss) and stopping_possible:
        patience += 1
        if patience == 5:
            utils.SaveModelMS(model,f'{folder}/earlystate_{params}.pt',M,S)
            best_early_loss = Val_loss[epoch]
            stopping_possible = False
    else :
        patience = 0
    prev_loss = Val_loss[epoch]


    if epoch == (50):
        for param_group in optimizer.param_groups :
            param_group['lr'] /= 10
        stopping_possible = True

    # if epoch == (190):
    #     for param_group in optimizer.param_groups :
    #         param_group['lr'] /= 10
    
    if epoch == (200):
        #save the model
        utils.SaveModelMS(model,f'{folder}/200state_{params}.pt',M,S)
        
    
torch.save(Training_loss,f'{folder}/training_loss_{params}.pt')
torch.save(Val_loss,f'{folder}/val_loss_{params}.pt')
utils.SaveModelMS(model,f'{folder}/finalstate_{params}.pt',M,S)
print(f'DONE, last val loss : {Val_loss[-1]:.7f}')
f.close()

# Path('LimitedHeston/LRlim').mkdir(exist_ok=True)
# g = open(f'LimitedHeston/LRlim/lr_{limit}_{version}.txt', 'a')
# g.write(f'lr = {lr:.4f}, val loss = {Val_loss[-1]:.4f}, early loss = {best_early_loss} \n')
# g.close()
