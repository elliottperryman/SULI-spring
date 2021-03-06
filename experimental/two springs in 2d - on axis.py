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


springs = [(0,1),(0,2)]


# In[ ]:


H, l, v, D = calc(R, springs)


# In[ ]:


plot(R, D[:3], .5)


# In[ ]:


plot(R, D[3:], .5)


# In[ ]:


# def Energy(R, springs, displacements):
#     return .5 * np.sum([ ((displacements[stop] - displacements[start]) @ ((R[stop]-start)/np.linalg.norm(R[stop]-R[start])))**2 for (start,stop) in springs])

def Energy(R, springs, displacements):
    E = 0
    for i, (start,stop) in enumerate(springs):
        v = R[stop] - R[start]
        v /= np.linalg.norm(v)
        
        stretch = displacements[stop] - displacements[start]
        
        E += .5 * (stretch @ v)**2
    return E


# In[ ]:


N,k = R.shape


# In[ ]:


H = np.zeros((N,k,N,k))


# In[ ]:


for (start,stop) in springs:
    v = R[stop] - R[start]
    v /= np.linalg.norm(v)
    v = v.reshape(-1,1)
    H[start,:,start,:] += v @ v.T
    H[stop,:,stop,:] += v @ v.T
    H[start,:,stop,:] -= v @ v.T
H = H.reshape(N*k,N*k)
H += H.T - np.eye(H.shape[0])*np.diag(H)


# In[ ]:


np.linalg.eigh(H)[0].round(2)


# In[ ]:


(H @ D[3].reshape(-1,1)).round(3)


# In[ ]:





# In[ ]:


Energy(R, springs, np.array([
    [1., 0],
    [0, 0],
    [0, 0]
]))


# In[ ]:


X = np.arange(-1,1,1e-2)
Y = np.arange(-1,1,1e-2)


# In[ ]:


Z = np.empty((len(X),len(Y)))


# In[ ]:


for i in range(len(X)):
    for j in range((len(Y))):
        Z[i,j] = Energy(R, springs, np.array([
            [X[i], Y[j]],
            np.zeros(2),
            np.zeros(2),
        ]))


# In[ ]:


plt.contourf(X,Y,Z.T); plt.colorbar();


# In[ ]:


for i in range(len(X)):
    for j in range((len(Y))):
        Z[i,j] = Energy(R, springs, np.array([
            np.zeros(2),
            [X[i], Y[j]],
            np.zeros(2),
        ]))


# In[ ]:


plt.contourf(X,Y,Z.T); plt.colorbar();


# In[ ]:


for i in range(len(X)):
    for j in range((len(Y))):
        Z[i,j] = Energy(R, springs, np.array([
            np.zeros(2),
            np.zeros(2),
            [X[i], Y[j]],
        ]))


# In[ ]:


plt.contourf(X,Y,Z.T); plt.colorbar();


# In[ ]:





# In[ ]:





# In[ ]:


for i in range(30):
    R = np.array([
        [0., 0],
        [1, i*.05],
        [-1, i*.05],
    ])

    H, l, v, D = calc(R, springs)
    
    print(l.round(4))
    
    print('displacement',0.05*i)
    plot(R, D[3:])


# In[ ]:




