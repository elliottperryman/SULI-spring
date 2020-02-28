#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('run', '../Calculating.ipynb')


# In[ ]:


get_ipython().run_line_magic('run', '../Plotting.ipynb')


# In[ ]:


R = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [1, 0, 0],
],dtype=np.float64)


# In[ ]:


springs = [(0,1),(0,2),(3,1),(3,2),(2,1)]


# In[ ]:


H, l, v, D = calc(R, springs)


# In[ ]:


plot(R, D[:6])


# In[ ]:


plot(R, D[6:])


# In[ ]:


l.round(3)


# In[ ]:


plot(R, v[l<.1])


# In[ ]:




