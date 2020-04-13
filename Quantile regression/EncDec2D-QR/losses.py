#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:25:54 2020

QUANTILE LOSS FOR MULTIPLE Q INPUTA

@author: rodrigo
"""
import torch
import torch.nn as nn

class QLoss(nn.Module):
    """
    Loss for 2D inputs with format NxQxCxTxS
    """
    def __init__(self,quantiles=[0.5]):
        super(QLoss, self).__init__()
        self.quantiles = quantiles
        
    def forward(self, x_pred, x_target):
        N,Q,C,T,S = x_pred.size()
        losses = []
        for i,q in enumerate(self.quantiles):
            res = x_target.sub(x_pred[:,i])
            losses.append(torch.max(q*res, (q-1)*res).mean())
        return torch.stack(losses,dim=0).mean()
