#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('run', '../Calculating.ipynb')


# In[ ]:


get_ipython().run_line_magic('run', '../Plotting.ipynb')


# In[ ]:


R = np.array([
    [0., 0],
    [1, 1],
    [-1, 1],
])


# In[ ]:


springs = [(0,1),(0,2)]


# In[ ]:


H, l, v, D = calc(R, springs)


# In[ ]:


l.round(3)


# In[ ]:


plot(R, D[:3], .5)


# In[ ]:


plot(R, D[3:], .5)


# In[ ]:


l.round(2)


# In[ ]:


[potential(R, springs, d.reshape(-1,2)).round(3) for d in D]


# In[ ]:


plot(R, v, .5)


# In[ ]:





# In[ ]:


potential(R, springs, D[-1].reshape(-1,2))


# In[ ]:


plt.scatter(np.arange(0,1,.01), [potential(R, springs, x*D[2].reshape(-1,2)) for x in np.arange(0,1,.01)])


# In[ ]:


plt.scatter(np.arange(0,1,.01), [potential(R, springs, x*D[-1].reshape(-1,2)) for x in np.arange(0,1,.01)])


# In[ ]:


plt.scatter(np.arange(0,1,.01), [potential(R, springs, x*D[-2].reshape(-1,2)) for x in np.arange(0,1,.01)])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sumBasis(R,D)


# In[ ]:





# In[ ]:




