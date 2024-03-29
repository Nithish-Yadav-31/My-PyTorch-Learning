#!/usr/bin/env python
# coding: utf-8

# ## Basic initialization using empty() method

# In[1]:


import torch

## 1-D vector
x = torch.empty(3)
print("1-D vector",x)


# In[2]:


## 2-D matrix
x1 = torch.empty(2,3)
print("\n 2-D matrix",x1)


# In[3]:


## 3-D matrix
x2 = torch.empty(2,2,3)
print("\n 3-D Matrix",x2)


# In[4]:


## 4-D matrix
x3 = torch.empty(2,2,3,3)
print(x3)


# ## Other methods of initialization

# In[5]:


y = torch.rand(2,4)
y


# In[6]:


y1 = torch.ones(3,3,4)
y1


# In[7]:


y2 = torch.zeros(2,5)
y2


# In[8]:


## Initialization with a different data type

y3 = torch.ones(3,4, dtype=int)
y3


# In[9]:


## Initialization with a different data type

y4 = torch.ones(3,4, dtype=torch.float)
y4


# In[10]:


## Initialization with a different data type

y5 = torch.ones(3,4, dtype=torch.double)
y5


# ## Printing size of tensors

# In[11]:


## Printing Size of tensor

print(y1.size())


# In[12]:


## Printing Size of tensor

print(x3.size())


# ## Conversion using tensor() method

# In[13]:


## Converting a list to a tensor
n = torch.tensor([1,7,3,6])
n


# ## Arithmetic Operations

# In[14]:


n1 = torch.rand(3,3)
n2 = torch.rand(3,3)


# In[15]:


## Addition 
print(n1 + n2)
print(torch.add(n1,n2))

## ELement-Wise addition
print(n1.add_(n2))


# In[16]:


## Subraction
print(n1 - n2)
print(torch.sub(n1, n2))


# In[17]:


## Multiplication

print(n1 * n2)
print(torch.mul(n1, n2))


# In[18]:


## Division

print(n1 / n2)
print(torch.div(n1, n2))


# ## Indexing Tensors

# In[19]:


print(n1, n1.shape)
print("\n first row",n1[0, :])
print("\n first column",n1[:, 0])


# ## View function to display the tensor

# In[25]:


n1.view(-1, 3)


# In[27]:


n1.view(-1, 1)


# In[28]:


n2.view(-1, 9)


# In[ ]:





# In[ ]:





# In[29]:


import numpy
import torch


# In[31]:


## Creating a new tensor
a = torch.ones(5)
a


# In[35]:


## Creating a new numpy array
b = a.numpy()
b


# In[36]:


## Adding one to all elements of tensor 
a.add(1)


# In[38]:


## printing numpy array after addition operation to tensor
print(b)

## Refer the following markdown


# If the add() function has an effect on the numpy array, then they both might be assigned to the same memory location which will make the changes to both the arrays.

# In[ ]:





# In[39]:


## Moving the data to the GPU then performing the operations
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device = device)
    y = torch.ones(5)
    y = y.to(device)
    z = x+y


# In[40]:


## Numpy can only handle tensors on CPU which causes the errror.
z.numpy()


# In[41]:


## Moving tensor back to CPU from GPU
z = z.to("cpu")


# In[42]:


## After moving the CPU now the tensor gets accessible by the numpy
z.numpy()


# In[ ]:





# In[44]:


## requires_grad -> tells pytorch that it will need to calculate the gradient to this tensor later in the optimization step
x = torch.ones(5, requires_grad=True)


# In[45]:


## Whenever we need to optimize a variable, we have to specify this
x

