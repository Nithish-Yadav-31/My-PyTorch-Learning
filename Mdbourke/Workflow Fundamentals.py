# %%
"""
# what_were_covering = {
    1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "making predictions and evaluating a model (inference)",
    5: "saving and loading a model",
    6: "putting it all together"
}
"""

# %%
import torch
from torch import nn 
import matplotlib.pyplot as plt 

torch.__version__

# %%
"""
## 1. Data (preparing and loading)

- Let's create our data as a straight line.

- We'll use linear regression to create the data with known parameters (things that can be learned by a model) and then we'll use PyTorch to see if we can build model to estimate these parameters using gradient descent.
"""

# %%
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step)
y = weight * X + bias

X[:10], y[:10]

# %%
"""
### Split data into training and test sets
![image.png](attachment:image.png)
"""

# %%
# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)

# %%
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14});

# %%
plot_predictions();

# %%
"""
## 2. Building a model
![image.png](attachment:image.png)
"""

# %%
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.weights * X + self.bias

# %%
"""
### Checking the contents of a PyTorch model
"""

# %%
torch.manual_seed(42)

model_0 = LinearRegression()

list(model_0.parameters())

# %%
# List named parameters 
model_0.state_dict()

# %%
"""
### Making predictions using torch.inference_mode()

- As the name suggests, torch.inference_mode() is used when using a model for inference (making predictions).

- torch.inference_mode() turns off a bunch of things (like gradient tracking, which is necessary for training but not for inference) to make forward-passes (data going through the forward() method) faster.
"""

# %%
with torch.inference_mode():
    y_preds = model_0(X_test)

# %%
# Check the predictions
print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

# %%
plot_predictions(predictions=y_preds)

# %%
y_test - y_preds

# %%
"""
- Those predictions look pretty bad...
- This make sense though when you remember our model is just using *random parameter values* to make predictions.
"""

# %%
"""

## 3. Train model
![image.png](attachment:image.png)
- Let's create a loss function and an optimizer we can use to help improve our model.

- Depending on what kind of problem you're working on will depend on what loss function and what optimizer you use.

- However, there are some common values, that are known to work well such as the SGD (stochastic gradient descent) or Adam optimizer. And the MAE (mean absolute error) loss function for regression problems (predicting a number) or binary cross entropy loss function for classification problems (predicting one thing or another).

![image-2.png](attachment:image-2.png)
"""

# %%
loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# %%
"""
## Creating the training loop
![image.png](attachment:image.png)

- Calculate the loss (loss = ...) before performing backpropagation on it (loss.backward()).
- Zero gradients (optimizer.zero_grad()) before stepping them (optimizer.step()).
- Step the optimizer (optimizer.step()) after performing backpropagation on the loss (loss.backward()).
![image-2.png](attachment:image-2.png)
"""

# %%
"""
## Creating the testing loop
![image.png](attachment:image.png)
![image-2.png](attachment:image-2.png)
"""

# %%
torch.manual_seed(42)

epochs = 100

train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):

    ## Set the training mode
    model_0.train()

    ## 1. Initializing the forward pass
    y_pred = model_0(X_train)

    ## 2. Calculate the loss (between predictions and ground truth)
    loss = loss_fn(y_pred, y_train)

    ## 3. Sets gradients of all model parameters to zero.
    optimizer.zero_grad()

    ## 4. Compute backward loss
    loss.backward()

    ## 5. Progress the Optimizer moving forward
    optimizer.step()

    model_0.eval()

    with torch.inference_mode():
        test_pred = model_0(X_test)

        test_loss = loss_fn(test_pred, y_test.type(torch.float))

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss}")

# %%
# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();

# %%
# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

# %%
"""
- This is the whole idea of machine learning and deep learning, **there are some ideal values that describe our data** and rather than figuring them out by hand, **we can train a model to figure them out programmatically.**
"""

# %%
"""
## 4. Making predictions with a trained PyTorch model (inference)
- Once you've trained a model, you'll likely want to make predictions with it.

- We've already seen a glimpse of this in the training and testing code above, the steps to do it outside of the training/testing loop are similar.

There are three things to remember when making predictions (also called performing inference) with a PyTorch model:

- Set the model in evaluation mode (model.eval()).
- Make the predictions using the inference mode context manager (with torch.inference_mode(): ...).
- All predictions should be made with objects on the same device (e.g. data and model on GPU only or data and model on CPU only).
"""

# %%
model_0.eval()

with torch.inference_mode():
    y_preds = model_0(X_test)

y_preds

# %%
plot_predictions(predictions=y_preds)

# %%
"""
## 5. Saving and loading a PyTorch model
![image.png](attachment:image.png)


### Saving a PyTorch model's state_dict()
The recommended way for saving and loading a model for inference (making predictions) is by saving and loading a model's state_dict().

Let's see how we can do that in a few steps:

- We'll create a directory for saving models to called models using Python's pathlib module.
- We'll create a file path to save the model to.
- We'll call torch.save(obj, f) where obj is the target model's state_dict() and f is the filename of where to save the model.
"""

# %%
from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 

# %%
## Checking if the model is saved inside the folder


# %%
"""
## Loading a saved PyTorch model's state_dict()

- Loading the saved model requires two important components which are
    - Creating a new instance of the model architecture.
    - The saved model weights
"""

# %%
loaded_model = LinearRegression()

loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# %%
# 1. Put the loaded model into evaluation mode
loaded_model.eval()

# 2. Use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_model_preds = loaded_model(X_test) # perform a forward pass on the test data with the loaded model

# %%
# Compare previous model predictions with loaded model predictions (these should be the same)
y_preds == loaded_model_preds

# %%


# %%
