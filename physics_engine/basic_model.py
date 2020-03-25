#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


class point(object):
    def __init__(self, x, y, z):
        x,y,z = np.float64([x,y,z])
        self.x = x
        self.y = y
        self.z = z
        
    def move(self, x, y, z):
        self.x += x
        self.y += y
        self.z += z
        
    def __sub__(self, other):
        x_diff = self.x-other.x
        y_diff = self.y-other.y
        z_diff = self.z-other.z
        pt = point(x_diff, y_diff, z_diff)
        return pt
    
    def __add__(self, other):
        x_diff = self.x+other.x
        y_diff = self.y+other.y
        z_diff = self.z+other.z
        pt = point(x_diff, y_diff, z_diff)
        return pt   
    
    def __mul__(self, scalar):
        x_diff = self.x*scalar
        y_diff = self.y*scalar
        z_diff = self.z*scalar
        pt = point(x_diff, y_diff, z_diff)
        return pt 
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def dist(self):
        a = np.array([self.x,self.y,self.z],
                     dtype=np.float64)
        return np.sqrt(np.sum(np.power(a,2)))

    def print(self):
        print(self.x.round(3), self.y.round(3), self.z.round(3))


# In[ ]:


class spring(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.relaxed_length = (end-start).dist()
        
    def force(self, startMove, endMove):
        newEnd = self.end + endMove
        newStart = self.start + startMove
        newDist = (newEnd-newStart).dist()
        force = newDist-self.relaxed_length
        forceVector = force*(newEnd-newStart)
        return forceVector, -1.*forceVector
        
    def print(self):
        self.start.print()
        print(' is connected to ')
        self.end.print()
        print('')


# In[ ]:


class tetrahedron(object):
    def __init__(self):
        locs = [
            (1, 1, 1),
            (1, -1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
        ]
        indices = [
            (0,1),(0,2),(0,3),
            (1,2),(1,3),
            (2,3),
        ]
        self.pts = [point(*x) for x in locs]
        self.springs = [spring(self.pts[x[0]],self.pts[x[1]]) for x in indices]
        
    def print(self):
        for pt in self.pts: pt.print()
        for s in self.springs: s.print()


# In[ ]:


first = tetrahedron()


# In[ ]:


(first.pts[0]-first.pts[1]).dist()


# In[ ]:


s = first.springs[0]


# In[ ]:


s.print()


# ### I should see the forces on the spring start and end behaving sensibly

# In[ ]:


p1 = point(0,0,0)
p2 = point(0,0,0)


# In[ ]:


s = spring(p1,p2)


# In[ ]:


zero = point(0,0,0)


# In[ ]:


move = point(0,1,0)


# In[ ]:


for z in s.force(zero, move): z.print()


# In[ ]:





# In[ ]:


move = point(1,2,4)


# In[ ]:


for z in s.force(zero, move): z.print()


# In[ ]:





# In[ ]:




