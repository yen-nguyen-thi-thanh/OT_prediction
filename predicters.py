#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from geomloss import SamplesLoss
from sklearn.metrics import roc_curve, roc_auc_score
from utils import *
import logging




class OTpreds():
    """
    'One parameter equals one predicted value' model (Algorithm 1. in the paper)

    Parameters
    ----------

    eps: float, default=0.01
        Sinkhorn regularization parameter.
        
    lr : float, default = 0.01
        Learning rate.

    opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
        Optimizer class to use for fitting.
        
    max_iter : int, default=10
        Maximum number of round-robin cycles for imputation.

    niter : int, default=15
        Number of gradient updates for each model within a cycle.

    batchsize : int, defatul=128
        Size of the batches on which the sinkhorn divergence is evaluated.

    n_pairs : int, default=10
        Number of batch pairs used per gradient update.

    tol : float, default = 0.001
        Tolerance threshold for the stopping criterion.

    weight_decay : float, default = 1e-5
        L2 regularization magnitude.

    order : str, default="random"
        Order in which the variables are imputed.
        Valid values: {"random" or "increasing"}.

    unsymmetrize: bool, default=True
        If True, sample one batch with no missing 
        data in each pair during training.

    scaling: float, default=0.9
        Scaling parameter in Sinkhorn iterations
        c.f. geomloss' doc: "Allows you to specify the trade-off between
        speed (scaling < .4) and accuracy (scaling > .9)"


    """
    def __init__(self, 
                 eps=0.01, 
                 lr=1e-2,
                 mu= 1e-5,
                 opt=torch.optim.RMSprop, 
                 niter=2000,
                 batchsize=128,
                 n_pairs=30,
                 noise=0.1,
                 scaling=.9,
                 tau = 1.,
                 option = '1-ball',
                 cost= None):
        self.eps = eps
        self.lr = lr
        self.mu = mu
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.cost = cost
        self.option = option
        self.tau = tau
        self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=scaling,  backend="tensorized", cost = cost)


    def fit_transform_update(self, X, Xp, verbose=True, report_interval=500, init = None, theta_true=None):
        """
        Imputes missing values using a batched OT loss

        Parameters
        ----------
        X : torch.DoubleTensor or torch.cuda.DoubleTensor
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned
            (e.g. with NaNs).

        mask : torch.DoubleTensor or torch.cuda.DoubleTensor
            mask[i,j] == 1 if X[i,j] is missing, else mask[i,j] == 0.

        verbose: bool, default=True
            If True, output loss to log during iterations.
        

        X_true: torch.DoubleTensor or None, default=None
            Ground truth for the missing values. If provided, will output a
            validation score during training, and return score arrays.
            For validation/debugging only.

        Returns
        -------
        X_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
            Imputed missing data (plus unchanged non-missing data).


        """

        X = X.clone()
        Xp = Xp.clone()
        
        M, d = X.shape
        N, d = Xp.shape

        n = min(M, N)
        option = self.option
        tau = self.tau
        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2**e
            if verbose:
                logging.info(f"Batchsize larger that half size = {n // 2}. Setting batchsize to {self.batchsize}.")

        mask = torch.isnan(Xp).double()

        if init is None: 
            imps = (0.5	 + self.noise * torch.rand(mask.shape).double())[mask.bool()]
        else:
            init0 = init.copy()
            imps = torch.Tensor(init0)

        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr, momentum = self.mu)

        if verbose:
            logging.info(f"batchsize = {self.batchsize}, epsilon = {self.eps:.4f}")

        if theta_true is not None:
            maes = np.zeros(self.niter)
            rmses = np.zeros(self.niter)
            aucs = np.zeros(self.niter)
            theta_np = theta_true.numpy()
            
        for i in range(self.niter):



            for _ in range(self.n_pairs):
                Xp_filled = Xp.detach().clone()
                Xp_filled[mask.bool()] = imps

                idx1 = np.random.choice(M, self.batchsize, replace=False)
                idx2 = np.random.choice(N, self.batchsize, replace=False)
    
                X_mini = X[idx1]
                Xp_mini = Xp_filled[idx2]
    
                loss = self.sk(X_mini, Xp_mini)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    ### Catch numerical errors/overflows (should not happen)
                    logging.info("Nan or inf loss")
                    break

            
            

            if option == '1-ball':
                tau_tmp = tau*(int(mask.sum()))
            elif option == '1-norm':
                tau_tmp = self.lr*tau*(int(mask.sum()))
            else:
                tau_tmp = tau

            imps_tmp = imps.clone()
            imps_tmp = imps_tmp*(imps_tmp>0)
            imps_tmp = imps_tmp*(imps_tmp<=1)+ 1*(imps_tmp>1)
            imps.data = prox(imps_tmp, tau=tau_tmp, option=option) # very important line


            if theta_true is not None:
                maes[i] = MAE(imps, theta_true, mask = None).item()
                rmses[i] = RMSE(imps, theta_true, mask = None).item()
                imps_np = imps.clone().detach().numpy()
                aucs[i] = roc_auc_score(theta_np, imps_np )

            if verbose and (i % report_interval == 0):
                if theta_true is not None:
                    logging.info(f'Iteration, learning rate {i}:\t Loss: {loss.item() / self.n_pairs:.4f}\t '
                                 f'Validation MAE: {maes[i]:.4f}\t' f'RMSE: {rmses[i]:.4f}\t'
                                 f'AUC: {aucs[i]:.4f}')
                else:
                    logging.info(f'Iteration {i}:\t Loss: {loss.item() / self.n_pairs:.4f}')

        Xp_filled = Xp.detach().clone()
        Xp_filled[mask.bool()] = imps
        imps_np = np.array(imps.clone().detach())
        
        if theta_true is not None:
            return imps_np, maes, rmses, aucs
        else:
            return imps_np




