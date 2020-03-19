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


springs = [(0,1,1),(0,2,1)]


# In[ ]:


H, l, v, D = calc(R, springs)


# In[ ]:


H.round(2)


# In[ ]:


plot(R, D[:3], .5)


# In[ ]:


plot(R, D[3:], .5)


# In[ ]:


plot(R, v, .5)


# In[ ]:


move =  np.array([0., -2, 0, 1, 0, 1])
move /= np.linalg.norm(move)
move = move.reshape(-1,1)
move.round(2)


# In[ ]:


(D @ move).round(2)


# In[ ]:


(v @ move).round(2)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sumBasis(R,D)


# In[ ]:





# In[ ]:




