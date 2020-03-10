#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


def visualize_normal_mode(geometry, eigenvector, scale=10):
    vec = eigenvector.reshape(-1, 3)
    trace2 = go.Cone(
        x=geometry[:, 0],
        y=geometry[:, 1],
        z=geometry[:, 2],
        u=vec[:, 0] * scale,
        v=vec[:, 1] * scale,
        w=vec[:, 2] * scale,
        sizemode="absolute", # "absolute"
        sizeref=2,
        anchor="tail"
    )
    return [trace2]


# In[ ]:


def visualize_normal_modes(geometry, eigenvectors, scale=10, cols=3):
    N3, N3 = eigenvectors.shape
    rows = int(N3/cols)
    if N3 % cols > 0:
        rows += 1
    specs = [[{'is_3d': True} for i in range(cols)]
             for j in range(rows)]
    fig = make_subplots(rows=rows, cols=cols, specs=specs)
    for row in range(rows):
        for col in range(cols):
            i = row * cols + col
            if i >= N3:
                continue
            traces = visualize_normal_mode(geometry, eigenvectors[:, i], scale)
            fig.add_trace(traces[0], row=row + 1, col=col + 1)
    fig.update_layout(scene_aspectmode='data')
    return fig


# In[ ]:


from math import ceil


# In[ ]:


def plot2d(R, v, scale=.5):
    numCols = ceil(len(v)/3)
    plt.subplots(numCols, 3, sharex=True, sharey=True, figsize=(15, 4))
    for i in range(len(v)):
        plt.subplot(numCols, 3, i+1)
        if np.allclose(v[i], np.zeros(v.shape[1])): continue
        plt.xlim((-1.5,1.5))
        plt.ylim((-1.5,1.5))
        for j in range(len(R)):
            x, y, = R[j]
            plt.arrow(x,y, v[i,j*2]*scale, v[i,j*2+1]*scale, width=scale*.1)        
    plt.show()


# In[ ]:


def plot(R, v, *args):
    if R.shape[1]==2:
        plot2d(R, v, *args)
    else:
        return visualize_normal_modes(R, v.T, *args)


# In[ ]:


from ipywidgets import interactive
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


def sumBasis(R, D, scale=.5):
    
    def plotC(**c):

        vSum = np.zeros(D.shape[1])
        for key, value in c.items():
            vSum += value*D[int(key)]

        plt.figure(2)
        for i in range(len(R)):
            plt.arrow(R[i,0],R[i,1],vSum[i*2]*scale,vSum[i*2+1]*scale, width=scale*.1)        
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        
    args = {}
    for i in range(len(D)):
        args[str(i)] = (-1.5, 1.5, .05)
        
    interactive_plot = interactive(plotC, **args)

    return interactive_plot

