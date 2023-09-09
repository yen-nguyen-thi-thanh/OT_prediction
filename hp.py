#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from geomloss import SamplesLoss
from predicters import OTpreds
from utils import *
from random import random
from math import log, ceil, floor, log10
from time import time, ctime
from load_data import load_data

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.debug("test")
torch.set_default_tensor_type('torch.DoubleTensor')


#space = { 'tau': hp.uniform('tau', 0.1, 1),    
#    'alpha' : hp.uniform('alpha', 1/10, 20)
#}


# can be called multiple times
def hyperband(z, zp, theta_true, space , max_iter = 81, eta = 3,  skip_last = 1, n_pairs = 30, noise = 0.15, batchsize = 128, init = None, dry_run = False ):
    
    #max_iter = 81 # maximum iterations per configuration
    #eta = 3         # defines configuration downsampling rate (default = 3)

    logeta = lambda x: log10( x ) / log10(eta )
    s_max = int(logeta(max_iter ))
    B = (s_max + 1 ) * max_iter

    results = []    # list of dicts
    counter = 0
    best_loss = np.inf
    best_counter = -1
    if init == None:
        init0 = init
    else:    
        init0 = init.copy()
    
    for s in reversed( range(s_max + 1 )):

        # initial number of configurations
        n = int( ceil( B / max_iter / (s + 1) * eta ** s ))	

        # initial number of iterations per config
        r = max_iter * eta ** ( -s )

        # n random configurations
        T = [get_params(space = space) for i in range(n)] 

        for i in range(( s + 1 ) - int( skip_last )):# changed from s + 1

            # Run each of the n configs for <iterations> 
            # and keep best (n_configs / eta) configurations

            n_configs = floor(n * eta ** ( -i ))
            n_iterations = int(r * eta ** ( i ))

            print("\n*** {} configurations x {:.1f} iterations each".format( 
                n_configs, n_iterations ))
            val_losses = []
            early_stops = []

            for t in T:
                
                counter += 1
                print ("\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format( 
                    counter, ctime(), best_loss, best_counter ))
                start_time = time()

                if dry_run:
                    result = { 'theta': random(), 'loss': random(), 'mae': random(),  'rmse': random(), 
                             'auc': random()}
                else:
                    params0 = t.copy()
                    params0.pop('wt')
                    params0.pop('alpha')
                    sk_imputer = OTpreds(n_pairs = n_pairs,  noise = noise, batchsize = batchsize, niter = int(n_iterations),**params0)
                    theta, maes, rmses, aucs = sk_imputer.fit_transform_update(z, zp, theta_true = theta_true, init = init0, verbose=False)
                    result = {'theta': theta, 'loss': rmses[-1], 'mae' : maes[-1], 'rmse': rmses[-1], 'auc': aucs[-1]}

                assert( type( result ) == dict )
                assert( 'loss' in result )

                seconds = int( round( time() - start_time ))
                print("\n{} seconds.".format( seconds ))

                loss = result['loss']
                val_losses.append( loss )

                early_stop = result.get( 'early_stop', False )
                early_stops.append( early_stop )

                # keeping track of the best result so far (for display only)
                # could do it be checking results each time, but hey
                if loss < best_loss:
                    best_loss = loss
                    best_counter = counter

                result['counter'] = counter
                result['seconds'] = seconds
                result['params'] = t
                result['iterations'] = n_iterations

                results.append( result )

            # select a number of best configurations for the next loop
            # filter out early stops, if any
            indices = np.argsort( val_losses )
            T = [ T[i] for i in indices if not early_stops[i]]
            T = T[ 0:int( n_configs /eta )]

    return results

