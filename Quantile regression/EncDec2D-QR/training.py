#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 8:15:27 2020

@author: rodrigo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import json
import numpy as np
import configargparse
from datetime import datetime
from collections import defaultdict
from tqdm import trange

from losses import QLoss
from utils import Logger, DotDict, prepare_datasets, rmse, bias, crps, wmape
from cnn import EncDec

import matplotlib.pyplot as plt
from plots import heatmap_example, plot_quantiles, plot_learning_curves, reliability

"""
#######################################################################################################################
# VARIABLES AND OPTIONS
#######################################################################################################################
"""
p = configargparse.ArgParser()
# -- data
p.add('--datadir', type=str, help='path to dataset', default='../data')
p.add('--dataset', type=str, help='dataset name', default='epoc')
p.add('--extension', type=str, help='dataset extension', default='.csv', choices=['.csv', '.pt', 'pickle'])
# -- exp
p.add('--outputdir', type=str, help='path to save exp', default='output/' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
# -- model
p.add('--t_inp', type=int, help='number of input timesteps', default=28)
p.add('--t_out', type=int, help='number of output timesteps', default=7)
p.add('--n_points', type=int, help='number of spatial points/sensors', default=33)
p.add('--n_inp', type=int, help='number of input features', default=1)
p.add('--n_out', type=int, help='number of output features', default=1)
# -- evaluation
p.add('--k', type=int, help='fold of k-cross-validation', default=9)
p.add('--norm', type=str, help='type of normalization', default='z', choices=['min-max', 'z', 'log'])
p.add('--sp_norm', type=bool, help='wether normalization is by spatial point or not', default=True)
# -- optim
p.add('--lr', type=float, help='learning rate', default=1e-4)
p.add('--lr_t', type=float, help='learning rate threshold for decay', default=1e-6)
p.add('--beta1', type=float, help='adam beta1', default=.9)
p.add('--beta2', type=float, help='adam beta2', default=.999)
p.add('--eps', type=float, help='adam eps', default=1e-8)
p.add('--wd', type=float, help='weight decay', default=1e-5)
# -- learning
p.add('--batch_size', type=int, default=256, help='batch size')
p.add('--patience', type=int, default=5, help='number of epoch to wait before trigerring lr decay')
p.add('--n_epochs', type=int, default=200, help='number of epochs to train for')
# p.add('--q', type=float, default=0.5, help='quantile to learn')
p.add('--q', type=float, default=[0.05,0.5,0.95], help='quantiles to learn')
# -- gpu
p.add('--device', type=int, default=0, help='-1: cpu; > -1: cuda device id')

# parse
opt = DotDict(vars(p.parse_args()))

if opt.device > -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device)
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

choices=['min-max', 'z', 'log']
if opt.norm not in choices:
    raise ValueError("Not a valid normalization technique")

# Median index
if 0.5 in opt.q:
    m_index = opt.q.index(0.5)
else:
    raise ValueError("Quantile 0.5 is mandatory")

opt.outputdir = opt.outputdir + '_' + opt.dataset + '_' + str(opt.k)



"""
#######################################################################################################################
# DATA PREPARATION
#######################################################################################################################
"""
if opt.extension == '.csv':
    data = torch.Tensor(np.genfromtxt(os.path.join(opt.datadir, opt.dataset + opt.extension)))
    # data = data[-(5*8760):]
elif opt.extension == '.pt':
    data = torch.load(os.path.join(opt.datadir, opt.dataset + opt.extension))
    data = data[:,-int(2*8760):]
elif opt.extension == 'pickle':
    import pandas as pd
    data = torch.Tensor(pd.read_pickle(os.path.join(opt.datadir, opt.dataset)).values)
    data = data[:17500]
    # import random
    # col_idxs = list(range(data.size(1)))
    # random.shuffle(col_idxs)
    # data = data[:, torch.tensor(col_idxs)]
else:
    raise ValueError("Not a valid file extension for data")

allowed_dim = [2,3,4]
if data.dim() not in allowed_dim:
    raise ValueError("Unknown spatio-temporal data format")


X_train, Y_train, X_val, Y_val, X_test, Y_test, sc = prepare_datasets(data, opt.t_inp, 
                                                                      opt.t_out, opt.n_points,
                                                                      opt.k, opt.n_inp,
                                                                      opt.norm, opt.sp_norm)


train_dataset = []
for i in range(len(X_train)):
    train_dataset.append([X_train[i], Y_train[i]])
    
val_dataset = []
for i in range(len(X_val)):
    val_dataset.append([X_val[i], Y_val[i]]) 
    
test_dataset = []
for i in range(len(X_test)):
   test_dataset.append([X_test[i], Y_test[i]]) 
    
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = opt.batch_size,
                                           shuffle = True)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                           batch_size = len(X_val),
                                           shuffle = False)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = len(X_test),
                                           shuffle = False)


"""
#######################################################################################################################
# MODEL
#######################################################################################################################
"""
model = EncDec(opt.t_inp, opt.t_out, opt.n_points, len(opt.q), opt.n_inp, opt.n_out, past_t=3, hidden_dim=50)
model = model.to(device)
    

"""
#######################################################################################################################
# OPTIMIZER
#######################################################################################################################
"""
loss_fn = QLoss(opt.q)

optimizer = optim.Adam(model.parameters(),  lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.wd)

if opt.patience > 0:
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.patience)


"""
#######################################################################################################################
# LOGGER
#######################################################################################################################
"""
logger = Logger(opt.outputdir, 25)
with open(os.path.join(opt.outputdir, 'config.json'), 'w') as f:
    json.dump(opt, f, sort_keys=True, indent=4)
    
logger.log('k', opt.k)
logger.log('norm', opt.norm)

"""
#######################################################################################################################
# TRAINING AND VALIDATION
#######################################################################################################################
"""
lr = opt.lr
tr = trange(opt.n_epochs, position=0, leave=True)

for t in tr: 
    model.train()
    logs_train = defaultdict(float)
    for i, (x,y) in enumerate(train_loader):
        if torch.cuda.is_available():
            x = x.to(device)
            y = y.to(device)
        else:
            x = x
            y = y
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        logs_train['qloss'] += loss.item()
        
    # Logs training
    logs_train['qloss'] /= (i+1)
    logger.log('train', logs_train)

    model.eval()
    logs_val = defaultdict(float)
    with torch.no_grad():
        for x,y in val_loader:
            if torch.cuda.is_available():
                x = x.to(device)
                y = y.to(device)
            else:
                x = x
                y = y
            y_pred = model(x)
            loss_val = loss_fn(y_pred, y)
            logs_val['qloss'] = loss_val.item()
            
            # Logs evaluation
            logger.log('val', logs_val)

    # General information
    tr.set_postfix(train_qloss=logs_train['qloss'], val_qloss=logs_val['qloss'],
                   lr = lr)
    logger.checkpoint(model)
    
    if opt.patience > 0:
        lr_scheduler.step(logs_val['qloss'])
        lr = optimizer.param_groups[0]['lr']
    if lr <= opt.lr_t:
        break

"""
#######################################################################################################################
# TEST
#######################################################################################################################
"""
model.eval()
logs_test = defaultdict(float)
with torch.no_grad():        
    for x,y in test_loader:
        if torch.cuda.is_available():
            x = x.to(device)
            y = y.to(device)
        else:
            x = x
            y = y
        y_pred = model(x)
        
        _,Q,_,_,_ = y_pred.size()
        y_pred_dnorm = []
        # For simplicity, we store predictions in QxNxCxTxS format
        for i in range(Q):
            y_pred_dnorm.append(sc.rev_scale(y_pred[:,i].cpu()).view(1,-1,opt.t_out, opt.n_points))
        y_pred_dnorm = torch.cat(y_pred_dnorm, dim = 0)
        y_dnorm = sc.rev_scale(y.cpu()).view(-1,opt.t_out, opt.n_points)

        loss_test = loss_fn(y_pred, y)
        
        logs_test['qloss'] = loss_test.item()
        logs_test['rmse'] = rmse(y_pred_dnorm[m_index], y_dnorm)
        logs_test['bias'] = bias(y_pred_dnorm[m_index], y_dnorm)
        logs_test['crps'] = crps(y_pred_dnorm, y_dnorm, opt.q)
        logs_test['wmape'] = wmape(y_pred_dnorm[m_index], y_dnorm)
        
        logger.log('test', logs_test)
        
print("\n\n================================================")
print(" *  Test QLoss: ", logs_test['qloss'],
      "\n *  Test RMSE: ", logs_test['rmse'],
      "\n *  Test Bias: ", logs_test['bias'],
      "\n *  Test CRPS: ", logs_test['crps'],
      "\n *  Test WMAPE (%): ", logs_test['wmape'])
print("================================================\n")
print("\nNormalization:" , logger.logs['norm'][0], ', k:', logger.logs['k'][0])

logger.save(model)

"""
#######################################################################################################################
# SOME VISUALIZATIONS
#######################################################################################################################
"""
value, index = torch.min(y_pred[:,m_index].sub(y).abs().mean((1,2,3)), dim = 0)
print('---- Best test example ----\nError:', value.item(), '\nIndex:', index.item())

ind = index
real = sc.rev_scale(Y_test[ind].unsqueeze(0)).view(opt.t_out, opt.n_points)
prediction = y_pred_dnorm[:,ind]

heatmap_example(real, prediction, m_index, opt.t_out, opt.n_points)
plot_quantiles(real, prediction, opt.t_out, 19, opt.q)
plot_learning_curves(logger)
reliability(y_dnorm, y_pred_dnorm, opt.t_out, opt.q)

# plt.hist(y_pred_dnorm[:,:,:,0].reshape(-1),50, cumulative=True, density=True, histtype='step', label = 'Predicted')
# plt.hist(y_dnorm[:,:,0].reshape(-1),50, cumulative=True, density=True, histtype='step', label = 'Real')
# plt.legend()

# data = torch.Tensor(np.genfromtxt(os.path.join(opt.datadir, opt.dataset + opt.extension)))
# data2 = data[-(1*8760):]
# X_train2, Y_train2, X_val2, Y_val2, X_test2, Y_test2, sc2 = prepare_datasets(data2, opt.t_inp, 
#                                                                       opt.t_out, opt.n_points,
#                                                                       9, opt.n_inp,
#                                                                       opt.norm, opt.sp_norm)

# test_dataset2 = []
# for i in range(len(X_train2)):
#     test_dataset2.append([X_train2[i], Y_train2[i]]) 

# test_loader2 = torch.utils.data.DataLoader(dataset = test_dataset2,
#                                             batch_size = len(X_train2),
#                                             shuffle = False)

# model.to('cpu')
# model.eval()
# with torch.no_grad():        
#     for x,y in test_loader2:
#         if torch.cuda.is_available():
#             x = x
#             y = y
#         else:
#             x = x
#             y = y
#         y_pred = model(x)

#         _,Q,_,_,_ = y_pred.size()
#         y_pred_dnorm = []
#         for i in range(Q):
#             y_pred_dnorm.append(sc2.rev_scale(y_pred[:,i].cpu()).view(1,-1,opt.t_out, opt.n_points))
#         y_pred_dnorm = torch.cat(y_pred_dnorm, dim = 0)
#         y_dnorm = sc2.rev_scale(y.cpu()).view(-1,opt.t_out, opt.n_points)

#         loss_test = loss_fn(y_pred, y)
#         print("\n=============================================")
#         print("   Test QLoss: ", loss_test.item(),
#               "\n   Test RMSE: ",rmse(y_pred_dnorm[m_index], y_dnorm),
#               "\n   Test Bias: ", bias(y_pred_dnorm[m_index], y_dnorm),
#               "\n   Test WMAPE(%): ", wmape(y_pred_dnorm[m_index], y_dnorm))
#         print("=============================================\n")

