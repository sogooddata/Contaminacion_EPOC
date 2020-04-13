
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

PRIMERA APROXIMACIÓN ENCODER-DECODER 

@author: rodrigo
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

"""
#######################################################################################################################
# ENCODER
#######################################################################################################################
"""
class Encoder(nn.Module):
    def __init__(self, n_inp, t_inp, n_points, past_t=3, hidden_dim=50):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.t_inp = t_inp
        self.n_points = n_points
        self.past_t = past_t
        
        # self.conv = nn.Conv2d(n_inp,hidden_dim*n_points,kernel_size=(past_t,n_points))
        self.convBlock1 = EncConvBlock(n_inp, n_points, past_t, hidden_dim,num_conv=1)
        
    def forward(self, x):
        xp = F.pad(x, pad = (0,0,self.past_t-1,0))
        # out = self.conv(xp)
        out = self.convBlock1(xp)
        return out.view(-1,self.hidden_dim, self.t_inp, self.n_points)
    
    
class EncConvBlock(nn.Module):
    def __init__(self, n_inp, n_points, past_t, hidden_dim, num_conv):
        super(EncConvBlock, self).__init__()
        features = [n_inp] + [hidden_dim*n_points for i in range(num_conv)]
        layers = []
        for i in range(len(features)-1):
            layers.append(nn.Conv2d(in_channels=features[i], out_channels=features[i+1], kernel_size=(past_t, n_points), bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i+1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
        self.op = nn.Sequential(*layers)
    def forward(self, x):
        return self.op(x)
    

"""
#######################################################################################################################
# DECODER
#######################################################################################################################
"""
class Decoder(nn.Module):
    def __init__(self, t_inp, t_out, n_points, n_quantiles, n_out=1, hidden_dim=50):
        super(Decoder, self).__init__()
        self.t_inp = t_inp
        self.t_out = t_out
        self.n_points = n_points
        self.n_quantiles = n_quantiles
        self.n_out = n_out
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(hidden_dim, n_out, kernel_size=(1,1))
        self.norm = nn.BatchNorm2d(num_features=n_out)
        self.relu = nn.ReLU()
        
        self.regressor = nn.Linear(t_inp*n_points*n_out, n_quantiles*t_out*n_points*n_out)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.regressor(out.view(-1, self.n_out*self.t_inp*self.n_points))
        return out.view(-1, self.n_quantiles, self.n_out, self.t_out, self.n_points)
    
"""
#######################################################################################################################
# MODEL
#######################################################################################################################
"""
    
class EncDec(nn.Module):
  def __init__(self,t_inp, t_out, n_points, n_quantiles, n_inp=1, n_out=1, past_t = 3, hidden_dim=50):
    super(EncDec, self).__init__()
    self.encoder = Encoder(n_inp, t_inp, n_points, past_t, hidden_dim)
    self.decoder = Decoder(t_inp, t_out, n_points, n_quantiles, n_out, hidden_dim)
  
  def forward(self, x):
      out = self.encoder(x)
      out = self.decoder(out)
      return out
      
    

