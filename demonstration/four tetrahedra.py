#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('run', '../calculating.ipynb')


# In[ ]:


get_ipython().run_line_magic('run', '../plotting.ipynb')


# These and ***pretty much the rest of the code*** is straight from Tess Smidt: [Tess's website](https://blondegeek.github.io)

# In[ ]:


R = np.array([
    [0., 0., 0.], [1., 1., 0], 
    [0., 1., 1.], [1., 0., 1.],
    [0., 1., -1.], [1., 0., -1.],
    [2., 1., -1.], [2., 0., 0.],
    [2, 1, 1],
])

springs = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), 
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 3),
    (3, 7), (3, 8),
    (4, 5),
    (5, 6), (5, 7),
    (6, 7),
    (7, 8)
]


# In[ ]:


H, l, v, D = calc(R, springs)


# In[ ]:


l.round(3)


# In[ ]:


plot(R, D[:6], 6)


# In[ ]:


plot(R, D[6:9], 6)


# In[ ]:




