# %%
"""
Topic	Contents

0. Architecture of a classification neural network	Neural networks can come in almost any shape or size, but they typically follow a similar floor plan.
1. Getting binary classification data ready	Data can be almost anything but to get started we're going to create a simple binary classification dataset.
2. Building a PyTorch classification model	Here we'll create a model to learn patterns in the data, we'll also choose a loss function, optimizer and build a training loop specific to classification.
3. Fitting the model to data (training)	We've got data and a model, now let's let the model (try to) find patterns in the (training) data.
4. Making predictions and evaluating a model (inference)	Our model's found patterns in the data, let's compare its findings to the actual (testing) data.
5. Improving a model (from a model perspective)	We've trained an evaluated a model but it's not working, let's try a few things to improve it.
6. Non-linearity	So far our model has only had the ability to model straight lines, what about non-linear (non-straight) lines?
7. Replicating non-linear functions	We used non-linear functions to help model non-linear data, but what do these look like?
8. Putting it all together with multi-class classification	Let's put everything we've done so far for binary classification together with a multi-class classification problem.
"""

# %%
"""
## 0. Architecture of a classification neural network
![image.png](attachment:image.png)
"""

# %%
"""
## 1. Make classification data and get it ready
"""

# %%
from sklearn.datasets import make_circles

n_samples = 1000

X,y = make_circles(n_samples, noise=0.03, random_state=42)

# %%
print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")

# %%
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0],  "X2": X[:, 1], "label":y
                        })

circles.head(10)

# %%
circles.label.value_counts()

# %%
import matplotlib.pyplot as plt 
plt.scatter(x = X[:, 0], y=X[:, 1], c = y, cmap=plt.cm.RdYlBu)

# %%
"""
The above is a famous problem which is called the toy dataset, which is to classify red and blue dots using the neural network
"""

# %%
"""
## 1.1 Input and Output shapes

- One of the most common errors in deep learning is shape erros.
- Mismatching the shapes of tensors and tensor operations with result in errors in models.
- What can be done?
     - Get familiar with the shape of the data that you're working with.
     - Always try checking the input and output shapes of the tensors.

"""

# %%
X.shape, y.shape

# %%
## We do have a shape mismatch in the above, 

# %%
X_sample = X[0]
y_sample = y[0]

print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

# %%
"""
## 1.2 Converting Data into Tensors and creating train and test splits
"""

# %%
import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X[:5], y[:5]

# %%


# %%


# %%
