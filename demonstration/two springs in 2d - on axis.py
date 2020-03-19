#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('run', '../Calculating.ipynb')


# In[ ]:


get_ipython().run_line_magic('run', '../Plotting.ipynb')


# In[ ]:


R = np.array([
    [0., 0],
    [1, 0],
    [-1, 0],
])


# In[ ]:


springs = [(0,1,1),(0,2,1)]


# In[ ]:


H, l, v, D = calc(R, springs)


# In[ ]:


plot(R, D[:3], .5)


# In[ ]:


plot(R, D[3:], .5)


# In[ ]:





# In[ ]:




