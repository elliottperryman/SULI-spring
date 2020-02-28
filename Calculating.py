#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


# calc inertia axes
def inertiaAxes3d(x,y,z):
    I = np.empty((3,3))

    I[0,0] = np.sum(y**2+z**2)
    I[1,1] = np.sum(x**2+z**2)
    I[2,2] = np.sum(x**2+y**2)

    I[0,1] = I[1,0] = -np.sum(y*x)
    I[0,2] = I[2,0] = -np.sum(z*x)
    I[1,2] = I[2,1] = -np.sum(y*z)

    return I

def inertiaAxes2d(x,y):
    I = np.empty((2,2))

    I[0,0] = np.sum(y**2)
    I[1,1] = np.sum(x**2)

    I[0,1] = I[1,0] = -np.sum(y*x)
    return I


# In[ ]:


def zeroModes3d(R,X):
    N = R.shape[0]
    D = np.empty((6, N*3))

    
    for i in range(N):
        for j in range(3):
            D[3,i*3+j] = np.dot(R[i],X[1])*X[j,2]-np.dot(R[i],X[2])*X[j,1]
            D[4,i*3+j] = np.dot(R[i],X[2])*X[j,0]-np.dot(R[i],X[0])*X[j,2]  
            D[5,i*3+j] = np.dot(R[i],X[0])*X[j,1]-np.dot(R[i],X[1])*X[j,0]
            
    for i in range(len(R)):
        D[:3,i*3:i*3+3] = np.eye(3)

    # if colinear
    badOnes = []
    for i in range(len(D)):
        if np.allclose(D[i],np.zeros_like(D[i])):
            badOnes.append(i)
    for i in range(len(badOnes)):
        index = badOnes[i]-i
        D = np.append(D[:index],D[index+1:],0)
            
        
    for i in range(len(D)):
        D[i] /= np.linalg.norm(D[i])
            
    return D


def zeroModes2d(R,X):
    N = R.shape[0]
    D = np.zeros((3, N*2))

    for i in range(N):
        for j in range(2):
            D[2,i*2+j] = np.dot(R[i],X[0])*X[j,1]-np.dot(R[i],X[1])*X[j,0]

    for i in range(len(R)):
        D[:2,i*2:i*2+2] = np.eye(2)

    for i in range(len(D)):
        D[i] /= np.linalg.norm(D[i])
    return D


# In[ ]:


def calc(R, springs, threshold=0.01):
    
    # get parameters
    N = R.shape[0]
    k = R.shape[1]
    
    # normalize R
    mySum = np.sum(R, 0)
    for i in range(N):
        R[i] = R[i] - mySum*1.0/N

    
    # calculate H
    H = np.zeros((N,k,N,k))
    for i, (start, stop) in enumerate(springs):
        # strength of spring in each dimen.
        proj = R[stop]-R[start]
        proj /= np.linalg.norm(proj)
        
        #proj *= -np.sign(proj)
        
        # force on x1 connected to x2
        #   is k(x2-x1)  
        H[start,:,start,:] += np.eye(k)*proj**2
        H[stop,:,stop,:] += np.eye(k)*proj**2
        H[start,:,stop,:] += -np.eye(k)*proj**2
        H[stop,:,start,:] += -np.eye(k)*proj**2
        
    H = H.reshape((N*k,N*k))
    
    # get eigenvalues
    l, v = np.linalg.eigh(H)
    v = v.T
    
    # move to translating and rotating frame
    if k==2:
        I = inertiaAxes2d(R[:,0], R[:,1])
    elif k==3:
        I = inertiaAxes3d(R[:,0], R[:,1], R[:,2])
    else:
        raise Exception('dimension not 2 or 3')
        
    I_prime, X = np.linalg.eigh(I)
    
    # calculate the translating and rotating frames
    if k==2:
        D = zeroModes2d(R,X) 
    elif k==3:
        D = zeroModes3d(R,X) 


    
    # create new basis that is moving and ortho
    for i in range(len(v)):

        remainder = remainder = v[i] - np.sum((D @ v[i]).reshape(-1,1) * D, 0)
        norm = np.linalg.norm(remainder)

        if norm>threshold:
            remainder /= norm
            v[i] = remainder
            D = np.append(D, remainder.reshape(1,-1), 0)
        else:
            pass
            #v[i] = np.zeros_like(v[i])


#     if not np.allclose(D @ D.T, np.eye(len(H))):
#         raise Exception("D is not ortho")

#    l_new, v_new = np.linalg.eigh((D.T @ H @ D)[6:,6:])

    return H, l, v, D


# In[ ]:




