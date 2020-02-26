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


H.round(3)


# In[ ]:


v.round(3)


# In[ ]:


D.round(3)


# In[ ]:


plot(R, v)


# In[ ]:


plot(R, D)


# In[ ]:



