#!/usr/bin/env python
# coding: utf-8

# ## Gradient Calculation with AutoGrad

# In[1]:


## Can be used for calculating gradients, which are essential for model optimization


# In[2]:


import torch


# In[3]:


x = torch.rand(3, requires_grad=True)
x


# In[4]:


y = x+2
y


# In[5]:


z = y*y*2
z


# In[6]:


z = z.mean()
z


# In[7]:


## Computing gradient 
z.backward()
print(x.grad)


# In[ ]:





# In[9]:


## Preventing Gradient history


# In[ ]:





# In[ ]:




