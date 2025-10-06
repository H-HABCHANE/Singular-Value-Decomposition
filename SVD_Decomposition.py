#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np


# In[26]:


A=np.array([[1,1],[1,0],[0,1]])
print(A)


# In[27]:


USV=np.linalg.svd(A)
U=USV[0]
S=USV[1]
V=USV[2]


# In[28]:




# Affichage des résultats
print("U =")
print(U)
print("----------------")
print("S =")
print(S)
print("----------------")
print("Vt =")
print(Vt)


# In[18]:


V = Vt.T
print("V =")
print(V)


# In[19]:


2/np.sqrt(6)


# In[ ]:




