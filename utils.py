#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np

from scipy import optimize
from hyperopt import hp
from hyperopt.pyll.stochastic import sample


def get_params(space):

    params = sample(space)
    handle_integers(params)

    pi = np.random.normal(loc = 0, scale = 0.5, size = 5)
    pi = np.exp(pi)
    pi = 74*pi/np.sum(pi)
    nbw = [13, 12, 23, 25]
    w = []
    alpha = params['alpha']

    for i in range(len(pi)):
        w += nbw[i]*[pi[i]*1/nbw[i]]

    cost = dissim(torch.tensor(w), alpha).costs
    params['cost'] = cost
    params['wt'] = pi
    params['tau']= params['tau']

    return params



### Proximal operators
def prox(X, tau, option = '1-ball'):
    """
    Proximal operator of tau*1-norm or indicator of 1-norm ball of radius tau
    Parameters
    ----------
    X : torch.DoubleTensor, shape (M)
        
    tau : float
    
    option : string, either None or '1-norm' or '1-ball'
        function $g$
    
    Returns
    -------
    prox : torch.DoubleTensor, shape (M)
    
    """
    X = X.clone()
    if option == '1-norm':
        tmp = (X-tau > 0)*abs(X-tau)
        out = (tmp < 1)*tmp + (tmp >= 1)    
    elif option == '1-ball':
        nrm = sum(abs(X))
        if nrm <= tau:
            out = X
        else:
            X_abs = abs(X)
            mu, _ = torch.sort(X_abs, descending=True)
            mu_cs = torch.cumsum(mu, dim=0) - tau
            test = torch.where(mu - mu_cs/(torch.arange(1, 1+mu.shape[0])) > 0)
            rho = int(max(test[0]))
            theta = mu_cs[rho]/(rho+1)
            out = abs(X_abs - theta)*((X_abs - theta)>0)
            out = torch.where(X < 0, -out, out)
            out = torch.where(out==0., torch.zeros(len(out)), out)
    else:
        out = X
    return out

def nanmean(v, *args, **kwargs):
    """
    A Pytorch version on Numpy's nanmean
    """
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


#### Quantile ######
def quantile(X, q, dim=None):
    """
    Returns the q-th quantile.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data.

    q : float
        Quantile level (starting from lower values).

    dim : int or None, default = None
        Dimension allong which to compute quantiles. If None, the tensor is flattened and one value is returned.


    Returns
    -------
        quantiles : torch.DoubleTensor

    """
    return X.kthvalue(int(q * len(X)), dim=dim)[0]


#### Automatic selection of the regularization parameter ####
def pick_epsilon(X, quant=0.5, mult=0.05, max_points=2000):
    """
        Returns a quantile (times a multiplier) of the halved pairwise squared distances in X.
        Used to select a regularization parameter for Sinkhorn distances.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data on which distances will be computed.

    quant : float, default = 0.5
        Quantile to return (default is median).

    mult : float, default = 0.05
        Mutiplier to apply to the quantiles.

    max_points : int, default = 2000
        If the length of X is larger than max_points, estimate the quantile on a random subset of size max_points to
        avoid memory overloads.

    Returns
    -------
        epsilon: float

    """
    means = nanmean(X, 0)
    X_ = X.clone()
    mask = torch.isnan(X_)
    X_[mask] = (mask * means)[mask]

    idx = np.random.choice(len(X_), min(max_points, len(X_)), replace=False)
    X = X_[idx]
    dists = ((X[:, None] - X) ** 2).sum(2).flatten() / 2.
    dists = dists[dists > 0]

    return quantile(dists, quant, 0).item() * mult


#### Accuracy Metrics ####
def MAE(X, X_true, mask = None):
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        MAE : float

    """
    if mask == None:
        return torch.abs(X - X_true).sum() / len(X)
    else:
        if torch.is_tensor(mask):
            mask_ = mask.bool()
            return torch.abs(X[mask_] - X_true[mask_]).sum() / mask_.sum()
        else: # should be an ndarray
            mask_ = mask.astype(bool)
            return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()



def RMSE(X, X_true, mask = None):
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        RMSE : float

    """
    if mask == None:
        return (((X - X_true)**2).sum() / len(X)).sqrt()
    else:
        if torch.is_tensor(mask):
            mask_ = mask.bool()
            return (((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum()).sqrt()
        else: # should be an ndarray
            mask_ = mask.astype(bool)
            return np.sqrt(((X[mask_] - X_true[mask_])**2).sum() / mask_.sum())

        
        

def handle_integers( params ):
    new_params = {}
    for k, v in params.items():
        if type( v ) == float and int( v ) == v:
            new_params[k] = int( v )
        else:
            new_params[k] = v

    return new_params    
    
    
nbw = [6, 5, 4, 4]    

#dissim(wt = [6, 5, 3, 1, 4], alpha = 1).costs(x, y)
class dissim():
    
    def __init__(self, wt = nbw, alpha = 1): 
        """
        wt: torch.tensor 
        alpha: a number
        """
        self.wt = wt
        self.alpha = alpha
        
    def costs(self, x, y):
        """
        x: torch.tensor, (B, N, D) or (N, D)

        y: torch.tensor, (B, M, D) or (M, D)

        Return: a cost matrix as (B, N, M) tensor
        """

        xw = x[:,:, :-1]
        theta_x = x[:,:, -1]
        yw = y[:,:, :-1]
        theta_y = y[:,:, -1]
        xx = xw*torch.sqrt(self.wt)
        yy = yw*torch.sqrt(self.wt)

        x_col = torch.unsqueeze(xx, 2)
        y_lin = torch.unsqueeze(yy, 1)
        xys = torch.sum(torch.square(x_col - y_lin), axis = 3)

        theta_x_col = torch.unsqueeze(theta_x, 2)
        theta_y_lin = torch.unsqueeze(theta_y, 1)
        thetas = torch.square(theta_x_col - theta_y_lin)
        
        z = xys + self.alpha*thetas
        
        return z
    
    
class dissim2():
    
    def __init__(self, alpha): 
        """
        alpha: a number
        """
        self.alpha = alpha
        
    def cost2(self, x, y):
        """
        x: torch.tensor, (B, N, D) or (N, D)

        y: torch.tensor, (B, M, D) or (M, D)

        Return: a cost matrix as (B, N, M) tensor
        """

        xx = x[:,:, :-1]
        theta_x = x[:,:, -1]
        yy = y[:,:, :-1]
        theta_y = y[:,:, -1]
        x_col = torch.unsqueeze(xx, 2)
        y_lin = torch.unsqueeze(yy, 1)
        xx_col = torch.sqrt(torch.sum(x_col**2, axis = 3))
        yy_lin = torch.sqrt(torch.sum(y_lin**2, axis = 3))
        xys = torch.abs(xx_col-yy_lin)

        theta_x_col = torch.unsqueeze(theta_x, 2)
        theta_y_lin = torch.unsqueeze(theta_y, 1)
        thetas = torch.square(theta_x_col - theta_y_lin)

        z = self.alpha*xys + thetas

        return z
