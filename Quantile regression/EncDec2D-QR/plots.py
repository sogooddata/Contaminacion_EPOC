#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 07:45:38 2020

PLOTS

@author: rodrigo
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def heatmap_example(real, pred, q_index, x_max, y_max, fig_name="Heatmap", fig_size=(14,4), dpi=80):
    plt.figure(fig_name, figsize=fig_size, dpi=dpi)

    plt.subplot(1, 3, 1)
    plt.imshow(real.T,  vmin = real.min(), vmax = real.max(), aspect='auto', cmap='jet')
    plt.xticks(np.arange(0, x_max, 4))
    plt.yticks(np.arange(0, y_max, 1))
    plt.xlabel('timestep')
    plt.colorbar()
    plt.title('Ground truth')
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred[q_index].T, vmin = real.min(), vmax = real.max(), aspect='auto', cmap='jet')
    plt.xticks(np.arange(0, x_max, 4))
    plt.yticks(np.arange(0, y_max, 1))
    plt.xlabel('timestep')
    plt.colorbar()
    plt.title('Enc-Dec prediction')
    
    plt.subplot(1, 3, 3)
    plt.imshow(real.sub(pred[q_index]).abs().T, aspect='auto', cmap='gray_r')
    plt.xticks(np.arange(0, x_max, 4))
    plt.yticks(np.arange(0, y_max, 1))
    plt.xlabel('timestep')
    plt.colorbar()
    plt.title('Absolute error')


def plot_quantiles(real, pred, x_max, point, quantiles, fig_name="Quantiles", fig_size=(12,6), dpi=80):
    plt.figure(fig_name, figsize=fig_size, dpi=dpi)
    plt.plot(real[:,point], linewidth=2, label = 'Ground truth')
    for i,quantile in enumerate(quantiles):
        line_w = 1
        line_s = '--'
        if quantile == 0.5:
            line_w = 2
            line_s = '-'
        plt.plot(pred[i,:,point], line_s, linewidth=line_w, label = 'Quantile '+str(quantile))  
    Q = len(quantiles)
    for i in range(int(Q/2)):
        plt.fill_between(np.arange(0,x_max), pred[i,:,point], pred[-(i+1),:,point], color = 'blue', alpha = 0.05*i + 0.05)
    plt.legend()
    plt.title('Enc-dec prediction')
    plt.show()

def plot_learning_curves(logger, fig_name="Learning curves", fig_size=(12,4), dpi=80):    
    plt.figure(fig_name, figsize=fig_size, dpi=dpi)
    plt.plot(logger.logs['train.qloss'], label = 'Train')
    plt.plot(logger.logs['val.qloss'], label = 'Validation')
    plt.legend()
    plt.show()
    
def reliability(real, pred, x_max, quantiles, fig_name="Reliability", fig_size=(6,6), dpi=80):
    plt.figure(fig_name, figsize=fig_size, dpi=dpi)
    total = real.nelement()
    reliability = {}
    for i,quantile in enumerate(quantiles):
        count = (real <= pred[i]).sum().item()
        reliability.update({quantile: count/total })
    plt.plot(list(reliability.values()), list(reliability.keys()), '-o', label = 'Reliability')
    plt.plot([0, 1], [0, 1], color = 'red')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid()
    plt.show()
        