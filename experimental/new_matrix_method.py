#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


import numpy as np


# In[ ]:


R = np.array([
    [-1., 1],
    [0, 0],
    [1, 1],
])


# In[ ]:


displ = np.sum(R,0)
R -= displ/R.shape[0]


# In[ ]:


springs = [(0,1), (1, 2)]


# In[ ]:


N, k = R.shape


# In[ ]:


H = np.zeros((N,k,N,k))

for i, (start_,stop_) in enumerate(springs):
    v = (R[stop_]-R[start_]).reshape(-1,1)
    for start, stop in [(start_,stop_),(stop_,start_)]:
        H[start,:,start,:] += v@v.T*np.sign(v[0,0])
        H[stop,:,start,:] += -v@v.T*np.sign(v[0,0])

H = H.reshape(N*k,N*k)


# In[ ]:


H.round(3)


# In[ ]:


get_ipython().run_line_magic('run', '../calculating.ipynb')


# In[ ]:


I = inertiaAxes2d(R[:,0], R[:,1])


# In[ ]:


I_prime, X = np.linalg.eigh(I)


# In[ ]:


D = zeroModes2d(R, X)


# In[ ]:


get_ipython().run_line_magic('run', '../plotting.ipynb')


# In[ ]:


plot(R, D)


# In[ ]:


l, v = np.linalg.eigh(H)


# In[ ]:


v = v.T


# In[ ]:


plot(R, v)


# In[ ]:


l.round(3)


# In[ ]:


for i in range(len(v)):
    tmp = v[i]
    for j in range(len(D)):
        tmp -= D[j]*(D[j]@tmp)
    norm = np.linalg.norm(tmp)
    if norm<1e-4: 
        print(i)
        continue
    else:
        D = np.append(D, (tmp/norm)[np.newaxis,:], 0)


# In[ ]:


plot(R, D)


# In[ ]:




