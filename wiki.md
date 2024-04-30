***
# Fundamentals

## Chapter 0: Visualizing Gradient Descent

### Recap:
1. Defining a **simple linear regression model**
2. Generating **synthesis data** for it
3. Performing a **train_validation split** on our dataset
4. **Randomly initializing the parameters** of our model
5. Performing a **forward pass**, that is **making predictions** using our model
6. Computing the **errors** associated with our **predictions**
7. **Aggregating** the errors into a **loss** (mean square error)
8. Learning that the **number of data point** used to compute the loss defines the kind of gradient descent we're using: **batch**, **mini-batch**, or **stochastic**
9. Visualizing an example of a **loss surface** and using its **cross-sections** to get the **loss curves** for individual parameters
10. Learning that a **gradient is partial derivative** and it represents **how much the loss changes if one parameter changes a little bit**
11. Computing the **gradients** for our model's parameters using **equations, code and geometry**
12. Learning that **larger gradients** correspond to **steeper loss curves**
13. Learning that **batchpropagation** is nothing more than **"chained"** gradient descent
14. Using the **gradients** and a **learning rate** to **update the parameters**
15. Comparing the **effects on the loss** of using the **low, high and very high learning rates**
16. Learning that loss curves for all parameters should be, ideally, **similarly steep**
17. Visualizing the effects of using a **feature with a larger range**, making the loss curve for the corresponding parameter much steeper
18. Using Scikit-Learn's **StandardScaler** to bring a feature to a reasonable range and thus making the **loss surface more bowl-shape** and its cross-sections **similar steep**
19. Learning that **preprocessing steps** like scaling should be applied **after the train-validation split** to prevent **leakage**
20. Figuring out that performing **all steps** (forward pass, loss, gradients, and parameter update) makes **one epoch**
21. Visualizing the **path of gradient descent** over many epochs and realizing it is heavily **dependent on the kind of gradient descent** used: batch, mini-batch, or stochastic
22. learning that there is a **trade-off** between the stable and smooth path of batch gradient descent and the fast and chaotic path of stochastic gradient descent, making the use of **mini-batch gradient descent a good compromise** between the other two

### Recap 1: Defining a **simple linear regression model**

```python
y = b + wx + e
```
- b: bias or intercept, tells us the expected average value of y when x is 0
- w: weight or slope, tells us how much y increases when x increase by one unit
- e: error or noise, tells us the error we cannot get rid of.

```python
salary = minimum wage + increase per year * years of experience + noise
```

### Recap 2: Generating **synthesis data** for it

```python
true_b = 1
true_w = 2
N = 100

# data generation
np.random.seed(42)
x = np.random.rand(N, 1)
e = (0.1 * np.random.randn(N, 1))
y = true_b + true_w * x + e
```

### Recap 3: Performing a **train_validation split** on our dataset

The split should **ALWAYS THE FIRST THING** to do, no pre-processing, notthing come before the split.

```python
# shuffle the indices
idx = np.arange(N)
np.random.shuffle(idx)

# uses the first 80% for train
train_idx = idx[:int(0.8*N)]
# uses the rest 20% for validation
val_idx = idx[int(0.8*N):]

# generate train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]
```

Remeber to **plot the dataset** afterward.

### Recap 4: **Randomly initializing the parameters** of our model

This is usually **Step 0** in a Training Loop

```python
# Step 0 - Initialize parameters "b" and "w" randomly
np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)

b, w
```

```python
[0.49671415] [-0.1382643]
```

### Recap 5: Performing a **forward pass**, that is **making predictions** using our model

This is usually **Step 1** in a Training Loop

```python
# Step 1 - Compute model's predicted output - forward pass
yhat = b + w * x_train
```

### Recap 6: Computing the **errors** associated with our **predictions**

This is usually **Step 2.1** in a Training Loop

```python
# Step 2
# Step 2.1, compute errors on every data points
error = yhat - y_train
```

### Recap 7: **Aggregating** the errors into a **loss** (mean square error)

This is usually **Step 2.2** in a Training Loop

```python
# Step 2.2 compute the loss using the errors
loss = (error ** 2).mean()

loss
```
```python
2.7421577700550976
```

### Recap 11: Computing the **gradients** for our model's parameters using **equations, code and geometry**

This is usually **Step 3** in a Training Loop

```python
# Step 3 - Computes gradients for both "b" and "w"
b_grad = 2 * error.mean()
w_grad = 2 * (x_train * error).mean()

b_grad, w_grad
```
```python
-3.044811379650508 -1.8337537171510832
```

### Recap 14: Using the **gradients** and a **learning rate** to **update the parameters**
Learning Rate is a import hyper-parameter

```python
lr = 0.1
```

This is usually **Step 4** in Training Loop
```python
# Step 4 - Updates parameters using gradients and learning rate
b = b - lr * b_grad
w = w - lr * w_grad

b, w
```

```python
[0.80119529] [0.04511107]
```

### Recap 18: Using Scikit-Learn's **StandardScaler** to bring a feature to a reasonable range and thus making the **loss surface more bowl-shape** and its cross-sections **similar steep**

In order to reduce the curve's steepness of "b" and "w", one way is to **Scaling / Standarizing / Normalizing** our dataset.

Remember to **ONLY FIT THE TRAINING SET** to the scaler
```python
scaler = StandardScaler(with_mean=True, with_std=True)

# Use ONLY TRAIN DATA to fit the scaler
scaler.fit(x_train)

# We now can use the already fit scaler to Transform
scaled_x_train = scaler.transform(x_train)
scaled_x_val = scaler.transform(x_val)
```

### Recap 20: Figuring out that performing **all steps** (forward pass, loss, gradients, and parameter update) makes **one epoch**

This is **Step 5** in Training Loop, which we repeat all the previous steps.

***
<hr style="border:2px solid blue">

## Chapter 1: A Simple Regression Problem

### Recap
1. Implementing a Linear Regression in Numpy using **gradient descent**
2. Creating **tensor** in PyTorch, sending them to a **device**, and making **parameters** out of them
3. Understanding PyTorch's main feature, **autograd**, to perform automatic differentiation using its associated properties and methods, like backward(), grad, zero_() and no_grad()
4. Visualizing the **dynamic computation graph** associated with a sequence of operation
5. Creating an **optimizer** to simultaneously update multiple parameters, using its step() and zero_grad() methods
6. Creating a **loss function** using PyTorch's corresponding higher-order function
7. Understanding Pytorch's **Module** class and creating your own models, implementing **\_\_init__()** and **forward()** methods, and making use of its built-in **parameters()** and **state_dict** methods
8. Transforming the original Numpy implementation into a **PyTorch** one using the elements above
9. Realizing the importance of including **model.train()** in the **training loop**
10. Implementing **nested** and **sequential** models using PyTorch's **layers**
11. Putting it all together into neatly organized code devided into **three distint parts**: **data preparation**, **model configuration** and **model training**

### Recap 1: 1. Implementing a Linear Regression in Numpy using **gradient descent**

We will re-implement what we do in the previous chapter

```python
# Step 0 - Initialized parameters "b" and "w" randomly
np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)

print(b, w)

# Set learning rate
lr = 0.1
# Define epochs
n_epochs = 1000

for epoch in range(n_epochs):
    # Step 1 - compute model's output prediction - forward pass
    yhat = b + w * x_train

    # Step 2 - Compute error and loss
    error = (yhat - y_train)
    loss = (error ** 2).mean()

    # Step 3 - Compute gradient for both "b" and "w"
    b_grad = 2 * error.mean()
    w_grad = 2 * (x_train * error).mean()

    # Step 4 - Update parameters using gradient and learning rate
    b = b - lr * b_grad
    w = w - lr * b_grad

print(b, w)
```

```python
# b and w after initialization
[0.49671415] [-0.1382643]

# b and w after our gradient descent
[1.02354094] [1.96896411]
```

### Recap 2: Creating **tensor** in PyTorch, sending them to a **device**, and making **parameters** out of them

Tensors are just higher-dimensional matrics.
![tensor](image.png)

```python
# define device
devie = 'cuda' if torch.cuda.is_available() else 'cpu'

# Convert our Numpy arrays data into PyTorch's tensor and send to device
x_train_tensor = torch.as_tensor(x_train).float().to(device)
y_train_tensor = torch.as_tensor(y_train).float().to(device)
```

```python
# We can change our random parameters "b" and "w" to device's tensor too, but the implementation is a bit different
torch.manual_seed(42)

b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
w = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

b, w
```
```python
tensor([0.1940], device='cuda:0', requires_grad=True)
tensor([0.1391], device='cuda:0', requires_grad=True)
```

### Recap 3: Understanding PyTorch's main feature, **autograd**, to perform automatic differentiation using its associated properties and methods, like backward(), grad, zero_() and no_grad()

**Autograd** is PyTorch's automatic differentiation package. Thanks to it, we don't need to worry about partial derivative, chain rule or anything like this.

```python
# Set Learning Rate
lr = 0.1

# Step 0 - Initialize parameters "b" and "w" randomly
torch.manual_seed(42)

b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
w = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

# Define epochs
n_epochs = 1000

for epoch in range(n_epochs):
    # Step 1: Compute model's predicted output - forward pass
    yhat = b + w * x_train_tensor

    # Step 2 - Compute error and loss
    error = (yhat - y_train)
    loss = (error ** 2).mean()

    # Step 3 - Compute gradients for both "b" and "w"
    loss.backward()

    # Step 4 - Update parameters using gradients and learning rate
    with torch.no_grad():
        b -= lr * b.grad
        w -= lr * w.grad
    b.grad_zero()
    w.grad_zero()
```

### Recap 4: Visualizing the **dynamic computation graph** associated with a sequence of operation
We can use make_dot(yhat) to plot the graph

![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)


### Recap 5: Creating an **optimizer** to simultaneously update multiple parameters, using its step() and zero_grad() methods

An **optimizer** takes the parameters we want to update, the learning rate we want to use, and performs the **update** through its step() method.

```python
# define a SGD optimizer
optimizer = optim.SGD([b, w], lr=lr)
...

for epoch in range(n_epochs):
    ...

    # Step 4 - Update parameters using gradients and learning rate
    optimizer.step()
    optimizer.zero_grad()
```

### Recap 6: Creating a **loss function** using PyTorch's corresponding higher-order function.

```python
# define MSE loss function
loss_fn = nn.MSELoss(reduction='mean')

...
for epoch in range(n_epochs):
    ...
    
    # Step 2 - Compute the Loss
    loss = loss_fn(yhat, y_train_tensor)
    ...
```

### Recap 7: Understanding Pytorch's **Module** class and creating your own models, implementing **\_\_init__()** and **forward()** methods, and making use of its built-in **parameters()** and **state_dict** methods

```python
class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        # Compute the outputs / predictions
        self.linear(x)
```

```python
model = ManualLinearRegression().to(device)

...
for epoch in range(n_epochs):
    model.train()

    # Step 1 - Compute model's predicted output - forward pass
    yhat = model(x_train_tensor)

    ...
```

### Recap 10: Implementing **nested** and **sequential** models using PyTorch's **layers**

For **straightforward models** that use **a series of built-in PyTorch models**, such as Linear, where the output of one is sequentially fed as an input to the nex, we can use a **Sequential** model

![alt text](image-4.png)

```python
model = nn.Sequential(
    nn.Linear(3, 5),
    nn.Linear(5, 1)
).to(device)
```

or 

```python
model = nn.Sequential()
model.add_module('layer1', nn.Linear(3, 5))
model.add_module('layer2', nn.Linear(5, 1))
model.to(device)
```

There are **MANY** different layers that can be used in PyTorch
- Convolutions Layers
- Pooling Layers
- Padding Layers
- Non-Linear Activations
- Normalization Layers
- Recurrent Layers
- Transformer Layers
- Linear Layers
- Dropout Layers
- Sparse Layers (embeddings)
- Vision Layers
- DataParallel Layers (multi-GPU)
- Flatten Layer

### Recap 11: Putting it all together into neatly organized code devided into **three distint parts**: **data preparation**, **model configuration** and **model training**

```python
# data preparation
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# convert numpy dataset to tensor
x_train_tensor = torch.to_tensor(x_train).float().to(device)
y_train_tensor = torch.to_tensor(y_train).float().to(device)
```

```python
# model configuration

# set learning rate
lr = 0.1

torch.manual_seed(42)

# define model
model = nn.Sequential(
    nn.Linear(1, 1)
).to(device)

# define loss function
loss_fn = nn.MSELoss(reduction='mean')

# define optimizer
optim = optim.SGD(model.parameters(), lr=lr)
```

```python
# model training

n_epochs = 1000

for epoch in range(n_epochs):
    # set model to TRAIN model
    model.train()

    # Step 1 - Computes model's predicted output - forward pass
    yhat = model(x_train_tensor)

    # Step 2 - Compute Loss
    loss = loss_fn(yhat, y_train_tensor)

    # Step 3 - Compute gradients for both "b" and "w"
    loss.backward()

    # Step 4 - Update parameters using loss function and learning rate
    optimizer.step()
    optimizer.zero_grad()
```
## Chapter 2: Rethinking the Training Loop

## Chapter 2.1: Going Classy

## Chapter 3: A Simple Classification Problem


***
# Computer Vision


***
# Sequences


***
# Natural Language Processing

