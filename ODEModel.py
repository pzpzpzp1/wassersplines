from Utils import InputMapping

import torch, pdb
from torch import nn
from torchdiffeq import odeint_adjoint as odeint


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class velocMLP(nn.Module):
    def __init__(self, in_features=3, hidden_features=512, hidden_layers=2,
                 out_features=2, sigmac=3, n_freq=70, tdiv=1,
                 incrementalMask=True, Tperiod=None):
        super(velocMLP, self).__init__()

        net = []
        imap = InputMapping(in_features, n_freq, sigma=sigmac,
                            tdiv=tdiv, incrementalMask=incrementalMask, Tperiod=Tperiod)
        self.imap = imap
        net.append(imap)
        net.append(nn.Linear(imap.d_out, hidden_features))
        for i in range(hidden_layers):
            net.append(nn.Tanh())
            net.append(nn.Linear(hidden_features, hidden_features))
        net.append(nn.Softplus())
        net.append(nn.Linear(hidden_features, out_features))
        net = nn.Sequential(*net)
        self.f = net

    def get_z_dot(self, t, z):
        """z_dot is parameterized by a NN: z_dot = NN(t, z(t))"""
        # pdb.set_trace()
        if t.dim() == 0:
            t = t.expand(z.shape[0], 1)
        else:
            t = t.reshape(z.shape[0], 1)
        tz = torch.cat((t, z), 1)
        z_dot = self.f(tz)
        return z_dot

    def getGrads(self, tz, getJerk = False):
        """
        tz: N (d+1)
        out: N d
        jacs:
        """
        tz.requires_grad_(True)
        N = tz.shape[0]
        dim = tz.shape[1]-1 # dimension
        batchsize = tz.shape[0]
        z = tz[:, 1:]
        t = tz[:, :1]
        out = self.get_z_dot(t, z)

        jacobians = torch.zeros(batchsize, dim, dim+1).to(tz)
        for i in range(dim):
            jacobians[:, i, :] = torch.autograd.grad(
                out[:, i].sum(), tz, create_graph=True)[0]
        
        # get Jerk. 3rd time deriv. Promotes constant rotation.
        Jerk = torch.zeros(N,dim)
        if getJerk:
            for i in range(dim):
                JerkTZ = torch.autograd.grad(jacobians[:, i, dim].sum(), tz, create_graph=True)[0]
                Jerk[:,i] = JerkTZ[:,dim]
        
        return out, jacobians[:, :, 0:dim], jacobians[:, :, dim:], Jerk

    def forward(self, t, z):
        """
        Calculate the time derivative of z.
        Parameters
        ----------
        t : torch.Tensor
            time
        z : torch.Tensor
            state
        Returns
        -------
        z_dot : torch.Tensor
            Time derivative of z.
        """
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            t.requires_grad_(True)
            z_dot = self.get_z_dot(t, z)
        return z_dot

class FfjordModel(torch.nn.Module):

    def __init__(self, in_features=3, hidden_features=512, hidden_layers=2,
                 out_features=2, sigmac=3, n_freq=70, tdiv=1,
                 incrementalMask=True, Tperiod=None):
        super(FfjordModel, self).__init__()
        self.modelshape = {'in_features': in_features,
                           'hidden_features': hidden_features,
                           'hidden_layers': hidden_layers,
                           'out_features': out_features,
                           'n_freq': n_freq,
                           'Tperiod': Tperiod,
                           }
        self.velfunc = velocMLP(
            in_features, hidden_features, hidden_layers, out_features, sigmac,
            n_freq, tdiv, incrementalMask, Tperiod)

    def save_state(self, fn='results/outcache/models/state.tar'):
        selfdict = self.state_dict()
        selfdict['modelshape'] = self.modelshape
        torch.save(selfdict, fn)

    def load_state(self, fn='state.tar'):
        # pdb.set_trace()
        self_dict = torch.load(fn)
        ms = self_dict.pop('modelshape')
        self.velfunc = velocMLP(
            ms['in_features'], ms['hidden_features'], ms['hidden_layers'],
            ms['out_features'], 5, ms['n_freq'], 5, True, ms['Tperiod'])
        self.load_state_dict(self_dict)
        self = self.to(device)

    def forward(self, z, integration_times=None, reverse=False):
        if integration_times is None:
            integration_times = torch.tensor([0., 1.]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)
        state = odeint(
            self.velfunc,  # Calculates time derivatives.
            z,  # Values to update.
            integration_times,  # When to evaluate.
            method='dopri5',  # Runge-Kutta
            atol=1e-5,  # Error tolerance
            rtol=2e-5,  # Error tolerance
        )
        return state


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
