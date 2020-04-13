#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: rodrigo
"""

import os
import json
import torch
import numpy as np
from collections import defaultdict

from sklearn.model_selection import KFold
import itertools

import scalers


def prepare_datasets(data, t_inp, t_out, n_points, k, n_inp, norm, sp_norm):
    """
    Preparación de los datos de entrada mediante k-cross validation sin repetición de ejemplos
    en ningún caso. Está pensado para spot-forecasting
    """
    # Nos aseguramos de que, si no hay variables exógenas, el formato de entrada sea el correcto.
    if data.dim() == 2:
        data.unsqueeze_(0)
    C,T,S = data.size()
    
    # Se define la estrategia de k-cross validation.
    kf = KFold(n_splits=10, shuffle=False)
    
    # # Se cogen todos los índices menos los últimos. Pero creo que esto es mejorable o innecesario
    #  SI ES NECESARIO
    n_data = T - 1 - (t_inp+t_out)
    
    # Se define que timesteps estarán en qué set. Nos interesan los timesteps, da igual
    # de que variable así que ponemos 0 porque siempre existirá.
    train_indexes, test_indexes = next(itertools.islice(kf.split(data[:,0:n_data][0]), k, None))
    if k == 0:
        train_indexes = train_indexes[(t_inp + t_out):]
    elif k == (kf.n_splits-1):
        train_indexes = train_indexes[:-(t_inp + t_out)]
    else:
        middle_index = test_indexes[0]
        rem_index = np.arange(middle_index-(t_inp+t_out), middle_index+(t_inp+t_out))
        train_indexes = np.delete(train_indexes, rem_index)
        
    train_indexes, val_indexes = train_indexes[:int(len(train_indexes)*0.9)], train_indexes[int(len(train_indexes)*0.9):]
    
    train = []
    val = []
    test = []
    for train_index in train_indexes:
        train.append(data.unsqueeze(0)[:,:,train_index:train_index+(t_inp+t_out)]) 
    for val_index in val_indexes: 
        val.append(data.unsqueeze(0)[:,:,val_index:val_index+(t_inp+t_out)])
    for test_index in test_indexes: 
        test.append(data.unsqueeze(0)[:,:,test_index:test_index+(t_inp+t_out)])
    
    train = torch.cat(train, dim=0)
    val = torch.cat(val, dim=0)
    test = torch.cat(test, dim=0)
    
    
    if norm == 'min-max':
        sc = scalers.MinMaxScaler(by_point = sp_norm)
        sc.fit(train)
        data_train = sc.scale(train)
        data_val = sc.scale(val)
        data_test = sc.scale(test)
    elif norm == 'z':
        sc = scalers.ZScaler(by_point = sp_norm)
        sc.fit(train)
        data_train = sc.scale(train)
        data_val = sc.scale(val)
        data_test = sc.scale(test)
    elif norm == 'log':
        sc = scalers.LogScaler()
        data_train = sc.scale(train)
        data_val = sc.scale(val)
        data_test = sc.scale(test)
  
    X_train, Y_train = torch.split(data_train, t_inp, dim = 2)
    X_val, Y_val = torch.split(data_val, t_inp, dim = 2)
    X_test, Y_test = torch.split(data_test, t_inp, dim = 2)
    
    Y_train = Y_train[:,0].unsqueeze(1)
    Y_val = Y_val[:,0].unsqueeze(1)
    Y_test = Y_test[:,0].unsqueeze(1)
    
    return (X_train.float(), Y_train.float(), X_val.float(), Y_val.float(), X_test.float(), Y_test.float(), sc)



def mae(x_pred, x_target, dim=0):
    """
    MAE calculation
    If dim == 0, overall MAE. 
    If dim == 1, by spatial point.
    If dim == 2, by timestep.
    """
    if dim == 0:
        return x_pred.sub(x_target).abs().mean().item()
    elif dim == 1:
        return x_pred.sub(x_target).abs().mean((0,1))
    elif dim == 2:
        return x_pred.sub(x_target).abs().mean((0,2))
    else:
        raise ValueError("Not a valid dimension")

def wmape(x_pred, x_target, dim=0):
    """
    WMAPE calculation
    If dim == 0, overall MAE. 
    If dim == 1, by spatial point.
    If dim == 2, by timestep.
    """
    if dim == 0:
        return 100*(mae(x_pred, x_target, dim = dim)/(x_target.abs().mean())).item()
    elif dim == 1:
        return 100*(mae(x_pred, x_target, dim = 1)/(x_target.abs().mean((0,1))))
    elif dim == 2:
        return 100*(mae(x_pred, x_target)/(x_target.abs().mean((0,2))))
    else:
        raise ValueError("Not a valid dimension")


def rmse(x_pred, x_target, dim=0):
    """
    RMSE calculation
    If dim == 0, overall MAE. 
    If dim == 1, by spatial point.
    If dim == 2, by timestep.
    """
    if dim == 0:
        return x_pred.sub(x_target).pow(2).mean().sqrt().item()
    elif dim == 1:
        return x_pred.sub(x_target).pow(2).mean((0,1)).sqrt().squeeze()
    elif dim == 2:
        return x_pred.sub(x_target).pow(2).mean((0,2)).sqrt().squeeze()
    else:
        raise ValueError("Not a valid dimension")

def bias(x_pred, x_target, dim=0):
    """
    Bias calculation
    If dim == 0, overall MAE. 
    If dim == 1, by spatial point.
    If dim == 2, by timestep.
    """
    if dim == 0:
        return x_pred.sub(x_target).mean().item()
    elif dim == 1:
        return x_pred.sub(x_target).mean((0,1))
    elif dim == 2:
        return x_pred.sub(x_target).mean((0,2))
    else:
        raise ValueError("Not a valid dimension")
        
def heaviside(x_pred, x_target):
    return (x_pred >= x_target).int()

def crps(x_pred, x_target, quantiles, dim=0):
    diff = x_pred[1:] - x_pred[:-1]
    quantiles = torch.tensor(quantiles).view(-1,1,1,1) 
    fs = sum(diff*((quantiles-heaviside(x_pred,x_target))[1:]**2))
    up = (x_target > x_pred[-1]).nonzero(as_tuple = True) 
    bottom = (x_target < x_pred[0]).nonzero(as_tuple = True) 
    fs[up] += (x_target[up] - x_pred[-1][up])*1
    fs[bottom] += (x_pred[0][bottom]-x_target[bottom])*1
    if dim == 0:
        return fs.mean().item()
    elif dim == 1:
        return fs.mean((0,1))
    elif dim == 2:
        return fs.mean((0,2))
    else:
        raise ValueError("Not a valid dimension")
    

def model_summary(model):
    print("model_summary")
    print()
    print("Layer_name"+"\t"*7+"Number of Parameters")
    print("="*100)
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    print("="*100)
    print(f"Total Params:{total_param}")       
    
def percentile(t: torch.tensor, q: float):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
    
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result
      
class DotDict(dict):
    """
    Dot notation access to dictionary attributes
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    
class Logger(object):
    """
    Log information through the process
    """
    def __init__(self, log_dir, chkpt_interval):
        super(Logger, self).__init__()
        os.makedirs(os.path.join(log_dir))
        self.log_path = os.path.join(log_dir, 'logs.json')
        self.model_path = os.path.join(log_dir, 'model.pth')
        self.logs = defaultdict(list)
        self.logs['epoch'] = 0
        self.chkpt_interval = chkpt_interval

    def log(self, key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                self.log('{}.{}'.format(key, k), v)
        else:
            self.logs[key].append(value)

    def checkpoint(self, model):
        if (self.logs['epoch'] + 1) % self.chkpt_interval == 0:
            self.save(model)
        self.logs['epoch'] += 1

    def save(self, model):
        with open(self.log_path, 'w') as f:
            json.dump(self.logs, f, sort_keys=True, indent=4)
        torch.save(model.state_dict(), self.model_path)