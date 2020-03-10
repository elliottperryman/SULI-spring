#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('run', '../Calculating.ipynb')
get_ipython().run_line_magic('run', '../Plotting.ipynb')


# ## One spring on axis

# In[ ]:


R = np.array([
    [0., 0],
    [1, 0],
])

springs = [(0,1)]


# In[ ]:


H, l, v, D = calc(R, springs)


# In[ ]:


plot(R, D, .5)


# ## One spring off axis

# In[ ]:


R = np.array([
    [0., 0],
    [1, 1],
])

springs = [(0,1)]


# In[ ]:


H, l, v, D = calc(R, springs)


# In[ ]:


plot(R, D, .5)


# ## Two springs on axis

# In[ ]:


R = np.array([
    [-1., 0],
    [0., 0],
    [1, 0],
])

springs = [(0,1), (1, 2)]


# In[ ]:


H, l, v, D = calc(R, springs)


# In[ ]:


plot(R, D, .5)


# ## Two springs moving off axis

# In[ ]:


for i, delta in enumerate(np.arange(0, 1, 0.05)):
    R = np.array([
        [-1., delta],
        [0., delta],
        [1, delta],
    ])

    springs = [(0,1), (1, 2)]

    H, l, v, D = calc(R, springs)
    
    print('delta of',delta.round(3))
    plot(R, D[3:], .5)


# ## Two springs in equilateral triangle

# In[ ]:


R = np.array([
    [-np.cos(np.pi/3), np.sin(np.pi/3)],
    [0., 0],
    [np.cos(np.pi/3), np.sin(np.pi/3)],
])

springs = [(0,1), (1, 2)]


# In[ ]:


H, l, v, D = calc(R, springs)


# In[ ]:


l.round(3)


# In[ ]:


plot(R, D, .5)


# ## Two springs at corners

# In[ ]:


R = np.array([
    [-1., 1],
    [0., 0],
    [1, 1],
])

springs = [(0,1), (1, 2)]


# In[ ]:


H, l, v, D = calc(R, springs)


# In[ ]:


plot(R, D, .5)


# In[ ]:


l.round(3)


# ## Two springs parallel

# In[ ]:


R = np.array([
    [0., 1],
    [0., 0],
    [0, 1],
])

springs = [(0,1), (1, 2)]


# In[ ]:


H, l, v, D = calc(R, springs)


# In[ ]:


plot(R, D, .5)


# In[ ]:


plot(R, v, .5)


# In[ ]:


l.round(3)


# ## Three springs in equilateral triangle

# In[ ]:


R = np.array([
    [-np.cos(np.pi/3), np.sin(np.pi/3)],
    [0., 0],
    [np.cos(np.pi/3), np.sin(np.pi/3)],
])

springs = [(0,1), (1, 2), (0, 2)]


# In[ ]:


H, l, v, D = calc(R, springs)


# In[ ]:


plot(R, D, .5)


# In[ ]:





# In[ ]:




