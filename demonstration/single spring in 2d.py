#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('run', 'Calculating.ipynb')


# In[ ]:


get_ipython().run_line_magic('run', 'Plotting.ipynb')


# In[ ]:


R = np.array([
    [0., 0],
    [1, 1],
])


# In[ ]:


springs = [(0,1)]


# In[ ]:


H, l, v, D = calc(R, springs)


# In[ ]:


plot(R, D[:3])


# In[ ]:


plot(R, D[3:])


# In[ ]:


def Energy(R, springs, displacements):
    E = 0
    for i, (start,stop) in enumerate(springs):
        v = R[stop] - R[start]
        v /= np.linalg.norm(v)
        
        stretch = displacements[stop] - displacements[start]
        
        E += .5 * (stretch @ v)**2
    return E


# In[ ]:


displacements = np.array([
    [-0.5, 0.5],
    [0.5, -0.5]
])
Energy(R, springs, displacements)


# In[ ]:




