import torch
import torch.nn as nn
import math

#with stochstic 'extra' cell
class LEMCellStoch(nn.Module):
    def __init__(self, ninp, nhid,dt):
        super(LEMCellStoch, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.dt = dt
        self.inp2hid = nn.Linear(ninp, 4 * nhid)
        self.hid2hid = nn.Linear(nhid, 3 * nhid)
        self.transform_z = nn.Linear(nhid, nhid)
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.nhid)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, y, z):
        transformed_inp = self.inp2hid(x)
        transformed_hid = self.hid2hid(y)
        i_dt1, i_dt2, i_z, i_y = transformed_inp.chunk(4, 1)
        h_dt1, h_dt2, h_y = transformed_hid.chunk(3, 1)
        ms_dt_bar = self.dt * torch.sigmoid(i_dt1 + h_dt1)#
        ms_dt = self.dt * torch.sigmoid(i_dt2 + h_dt2)#

        z = (1.-ms_dt) * z + ms_dt * torch.tanh(i_y + h_y)
        y = (1.-ms_dt_bar)* y + ms_dt_bar * torch.tanh(self.transform_z(z)+i_z)

        return y, z


class LEMStoch(nn.Module):
    def __init__(self, ninp, nhid, nout,ycoeff,zcoeff,version = 0,dt=1.):
        super(LEMStoch, self).__init__()
        self.nhid = nhid
        self.cell = LEMCellStoch(ninp,nhid,dt)
        self.stoch_linear = nn.Linear(2*nhid,2*nhid)
        self.classifier = nn.Linear(nhid, nout)
        self.ycoeff = ycoeff
        self.zcoeff = zcoeff
        self.version = version
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)

    def forward(self, input, dt = None):
        ## initialize hidden states
        y = input.data.new(input.size(1), self.nhid).zero_()
        z = input.data.new(input.size(1), self.nhid).zero_()
        y_hidds = []
        for t,inp in enumerate(input):
            delta = dt[t]
            x =torch.cat((inp,delta),dim =-1)
            y, z = self.cell(x,y,z)
            Stoch = self.stoch_linear(torch.randn(x.size(0),2*self.nhid).to(next(self.parameters()).device))
            if self.version == 0 :
                noise_y, noise_z = (torch.tanh(Stoch)*delta).chunk(2,dim = 1)
            elif self.version == 1:
                noise_y, noise_z = (torch.tanh(Stoch*torch.sqrt(delta))).chunk(2,dim = 1)
            elif self.version == 2:
                noise_y, noise_z = (torch.tanh(Stoch)).chunk(2,dim = 1)
            elif self.version == 4:
                noise_y, noise_z = (torch.tanh(Stoch[0,0:2])).chunk(2)
            else :
                raise ValueError("version must be 0,1 or 2")
            y = y + self.ycoeff*noise_y#try with and without dt maybe
            z = z + self.zcoeff*noise_z
            y_hidds.append(y)
            # for eval run monte-carlo
        out = self.classifier(torch.stack((y_hidds), dim=0))
        return out





class LSTM(nn.Module) :
    def __init__(self, ninp, nhid, nout,ycoeff = 0,zcoeff = 0):
        super(LSTM, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.lstm = nn.LSTMCell(ninp, nhid)
        self.classifier = nn.Linear(nhid, nout)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)
    
    def forward(self, input,dt = None,limit = 1):
        ## initialize hidden states
        h_0 = input.data.new(input.size(1), self.nhid).zero_()
        c_0 = input.data.new(input.size(1), self.nhid).zero_()
        y_hidds = []
        for t,inp in enumerate(input):
            delta = dt[t]
            x =torch.cat((inp,delta),dim =-1)
            (h_0, c_0) = self.lstm(x, (h_0, c_0))
            y_hidds.append(h_0)
        out = self.classifier(torch.stack((y_hidds), dim=0))
        return out

class LEMCellODE(nn.Module):
    def __init__(self, ninp, nhid):
        super(LEMCellODE, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.inp2hid = nn.Linear(ninp, 4 * nhid)
        self.hid2hid = nn.Linear(nhid, 3 * nhid)
        self.transform_z = nn.Linear(nhid, nhid)
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.nhid)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, y, z,dt):
        transformed_inp = self.inp2hid(x)
        transformed_hid = self.hid2hid(y)
        i_dt1, i_dt2, i_z, i_y = transformed_inp.chunk(4, 1)
        h_dt1, h_dt2, h_y = transformed_hid.chunk(3, 1)
        epsilon = 1e-44
        ms_dt_bar = dt * torch.sigmoid(i_dt1 + h_dt1)
        ms_dt = dt * torch.sigmoid(i_dt2 + h_dt2)

        
        z = (1.-ms_dt) * z + ms_dt * torch.tanh(i_y + h_y)
        y = (1.-ms_dt_bar)* y + ms_dt_bar * torch.tanh(self.transform_z(z)+i_z)
        return y, z

class LEMODE(nn.Module):
    def __init__(self, ninp, nhid, nout,ycoeff=1,zcoeff=1):
        super(LEMODE, self).__init__()
        self.nhid = nhid
        self.cell = LEMCellODE(ninp,nhid)
        self.stoch_linear = nn.Linear(2*nhid,2*nhid)
        self.classifier = nn.Linear(nhid, nout)
        self.ycoeff = ycoeff
        self.zcoeff = zcoeff
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)

    def forward(self, input, dt = None,limit = 0.1):
        ## initialize hidden states
        y = input.data.new(input.size(1), self.nhid).zero_()
        z = input.data.new(input.size(1), self.nhid).zero_()
        y_hidds = []
        for t,inp in enumerate(input):
            x = inp
            delta = dt[t]
            divider = math.ceil(delta.max() / limit)
            delta = delta / divider
            assert (delta.max() <= limit)
            for extratimesteps in range(divider) :
                y,z = self.cell(x,y,z,delta)
                x = self.classifier(y) #predited next state to be used as input for next timestep
                Stoch = self.stoch_linear(torch.randn(2*self.nhid).to(next(self.parameters()).device))
                noise_y, noise_z = torch.tanh(Stoch).chunk(2)
                y = y + self.ycoeff*noise_y*delta
                z = z + self.zcoeff*noise_z*delta
            y_hidds.append(y)
            # for eval run monte-carlo
        out = self.classifier(torch.stack((y_hidds), dim=0))
        return out

class LEMCellODE2(nn.Module):
    def __init__(self, ninp, nhid):
        super(LEMCellODE2, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.inp2hid = nn.Linear(ninp, 4 * nhid)
        self.hid2hid = nn.Linear(nhid, 3 * nhid)
        self.transform_z = nn.Linear(nhid, nhid)
        self.transform_dt = nn.Linear(1,2)
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.nhid)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, y, z,dt):
        transformed_inp = self.inp2hid(x)
        transformed_hid = self.hid2hid(y)
        i_dt1, i_dt2, i_z, i_y = transformed_inp.chunk(4, 1)
        h_dt1, h_dt2, h_y = transformed_hid.chunk(3, 1)
        t_dt1, t_dt2 = self.transform_dt(dt).chunk(2,1)
        ms_dt_bar = (torch.sigmoid(t_dt1)*torch.sigmoid(i_dt1 + h_dt1))
        ms_dt =     (torch.sigmoid(t_dt2)*torch.sigmoid(i_dt2 + h_dt2))

        
        z = (1.-ms_dt) * z + ms_dt * torch.tanh(i_y + h_y)
        y = (1.-ms_dt_bar)* y + ms_dt_bar * torch.tanh(self.transform_z(z)+i_z)
        return y, z

class LEMODE2(nn.Module):
    def __init__(self, ninp, nhid, nout,ycoeff=1,zcoeff=1):
        super(LEMODE2, self).__init__()
        self.nhid = nhid
        self.cell = LEMCellODE2(ninp,nhid)
        self.classifier = nn.Linear(nhid, nout)
        self.stoch_linear = nn.Linear(2*nhid,2*nhid)
        self.ycoeff = ycoeff
        self.zcoeff = zcoeff
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)

    def forward(self, input,dt = None,limit = 0.1):
        ## initialize hidden states
        y = input.data.new(input.size(1), self.nhid).zero_()
        z = input.data.new(input.size(1), self.nhid).zero_()
        y_hidds = []
        for t,inp in enumerate(input):
            x = inp
            delta = dt[t]
            divider = math.ceil(delta.max() / limit)
            delta = delta / divider
            # assert (delta <= limit)
            for extratimesteps in range(divider) :
                y, z = self.cell(x,y,z,delta)
                #afterwards no new info so x=0
                x = self.classifier(y)
                Stoch = self.stoch_linear(torch.randn(x.size(0),2*self.nhid).to(next(self.parameters()).device))
                noise_y, noise_z = torch.tanh(Stoch).chunk(2,dim = 1)
                y = y + self.ycoeff*noise_y*delta
                z = z + self.zcoeff*noise_z*delta
            y_hidds.append(y)
            # for eval run monte-carlo
        out = self.classifier(torch.stack((y_hidds), dim=0))
        return out
