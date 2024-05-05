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
![tensor](md_images\image.png)

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

![alt text](md_images\image-1.png)
![alt text](md_images\image-2.png)
![alt text](md_images\image-3.png)


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

![alt text](md_images\image-4.png)

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

<hr style="border:2px solid blue">

***

## Chapter 2: Rethinking the Training Loop
### Recap
1. Writing a **higher-order function** that builds functions to perform **training steps**
2. Understanding PyTorch's **Dataset** and **TensorDataset** classes, implementing its **\_\_init__()**, **\_\_getitem__()**, and **\_\_len()** methods
3. Modifying our **training loop** to incorporate **mini-batch inner loop**
4. Using PyTorch's **random_split()** method to generate training and validation datasets
5. Writing a **higher-order function** that builds functions to perform **validation steps**
6. Realizing the **importance** of including **model.eval()** inside the validation loop
7. Remembering the purpose of **no_grad()** and using it to **prevent** any kind of **gradient computation during validation**
8. Using **SummaryWriter** to **interface** with TensorBoard for logging
9. Adding a graph representing our model to **TensorBoard**
10. Sending scalars to TensorBoard to track the **evolution of training and validation losses**
11. **Saving / checkpointing** and **loading** models to and from disk to allow **resuming model training** or **deployment**
12. Realizing the importance of **setting the mode** of the model: **train()** or **eval()**, for **checkpointing** or **deploying** for prediction, respectively.


### Recap 1: Writing a **higher-order function** that builds functions to perform **training steps**

The higher-order function that builds a training step function for us is taking the key elements of our training loop: **model**, **loss** and **optimizer**. The actual training step function to be returned will have two arguments, **features** and **labels**, and will return the corressponding **loss value**.

```python
def make_train_step_fn(model, loss_fn, optimizer):
    # Builds function to performs a step in the train loop
    def perform_train_step_fn(x, y):
        # Sets model to TRAIN mode
        model.train()

        # Step 1 - Compute model's predicted output - forward pass
        yhat = model(x)

        # Step 2 - Compute loss
        loss = loss_fn(yhat, y)

        # Step 3 - Compute parameters gradient
        loss.backward()

        # Step 4 - Update parameters
        optimizer.step()
        optimizer.zero_grad()

        # Return the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return perform_train_step_fn
```

```python
# model configuration

...

# Creates the train_step function for our model, loss function and optimizer
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
```

```python
# model training
...

n_epochs = 1000

losses = []

for epoch in range(n_epochs):
    # Perform one train step and returns the corresponding loss
    loss = train_step_fn(x_train_tensor, y_train_tensor)

    losses.append(loss)
```

### Recap 2: Understanding PyTorch's **Dataset** and **TensorDataset** classes, implementing its **\_\_init__()**, **\_\_getitem__()**, and **\_\_len()** methods

In PyTorch, a **dataset** is represented by a regular **Pyton class** that inherits from the **Dataset** class. You can think it as a list of tuples, each tuple corresponding to **one point (features, label)**

The most fundamental methods it needs to implement are:
- **\_\_init__(self)**: This takes **whatever arguments** are needed to build a **list of tuples** - it may be the name of a CSV file, two tensors or anything else
- **\_\_getiten__(self, index):** This allows the dataset to be **indexed** so that it can work like a list **(datset[i])** - it must return a **tuple (features, label)**. We can load the data **on demand**
- **\_\_len__(self)**: This simply returns the **size** of the whole dataset so, whenever it is sampled, its indexing is limited to the actual size

```python
class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[indx], self.y[index])

    def __len__(self):
        return len(self.x)

train_data = CustomDataset(x_train_tensor, y_train_tensor)

train_data[0]
```
```python
(tensor([0.7713]), tensor([2.4745]))
```

Otherwise, we can use PyTorch's **TensorDataset** class, which will do pretty much the same thing as our custom dataset above.

```python
train_data = TensorDataset(x_train_tensor, y_train_tensor)
train_data[0]
```
```python
(tensor([0.7713]), tensor([2.4745]))
```
### Recap 3: Modifying our **training loop** to incorporate **mini-batch inner loop**

Until now, we have used the **whole training data** at every training step. It has been **batch gradient descent** all along. But if we want to get serious, we **must** use **mini-batch** gradient descent. So we use PyTorch's **DataLoader** for this job. We tell it which **dataset** to use, the desired **mini-batch size**, and if we'd like to **shuffle** it or not. The batch size is usually a **powers of two**, such as 16, 32, 64, ...

Remember to set shuffle to True for training set to increase performance of gradient descent.

```python
train_loader = DataLoader(
    dataset=train_data,
    batch_size=16,
    shuffle=True
)
```

```python
# view a batch data
next(iter(train_loader))
```

From now on, it is very unlikely that you'll ever use a full batch gradient descent. So it makes sense to organize a piece of code that's going to be use repeatedly into it own function: the **mini batch inner loop**

```python
def mini_batch(device, data_loader, step_fn):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        mini_batch_loss = step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)
    
    loss = np.mean(mini_batch_losses)
    return loss.
```

```python
# model training

n_epochs = 200

losses = []

for epoch in range(n_epochs):
    # inner loop
    loss = mini_batch(device, train_loader, train_step_fn)
    losses.append(loss0)
```

### Recap 4: Using PyTorch's **random_split()** method to generate training and validation datasets


```python
# data preparation

# convert Numpy to tensor
x_tensor = torch.as_tensor(x).float().to(device)
y_tensor = torch.as_tensor(x).float().to(device)

# build dataset containing all data point
dataset = TensorDataset(x_tensor, y_tensor)

# perform the split
ratio = 0.8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_trail
train_data, val_data = random_split(dataset, [n_train, n_val])

# build loader for each set
train_loader = DataLoader(
    dataset=train_data,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_data,
    batch_size=16
)
```

### Recap 5: Writing a **higher-order function** that builds functions to perform **validation steps**

We need to also set up a higher-order function to perform validation step in order to evaluate our model, using **eval** mode. The function will input our model and loss function, but not the optimizer, since it is not updating our parameters.

```python
def make_val_step_fn(model, loss_fn):
    # builds function that perform a step in the validation loop
    def perform_val_step_fn(x, y):
        # set model to eval mode
        model.evel()

        # Step 1 - Compute model's predicted output - forward pass
        yhat = model(x)

        # Step 2 - Compute loss
        loss = loss_fn(yhat, y)

        return loss.item()

    return perform_val_step_fn
```

```python
# model configuration

...

# create the val_step function for our model and loss function
val_step_fn = make_val_step_fn(model, loss_fn)
```

```python
# model training

n_epochs = 200

losses = []
val_losses = []

for epoch in range(n_epochs):
    # inner loop
    loss = mini_batch(device, train_loader, train_step_fn)
    losses.append(loss)

    # validation
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses.append(val_loss)
```

### Recap 11: **Saving / checkpointing** and **loading** models to and from disk to allow **resuming model training** or **deployment**

To checkpoint a model, we basically save its **state** to a file so that it can be **loaded** back later. Model's state is defined by:
- model.state_dict()
- optimizer.state_dict()
- losses
- epoch
- anything else you'd like to have restored later

```python
checkpoint = {
    'epoch': n_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': losses,
    'val_loss': val_losses
}

torch.save(checkpoint, 'model_checkpoint.pth')
```

```python
checkpoint = torch.load('model_checkpoint.pth')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

saved_epoch = checkpoint['epoch']
saved_loss = checkpoint['loss']
saved_val_loss = checkpoint['val_loss']

# always use TRAIN to resuming training
model.train() 
```

```python
# make prediction
new_inpute = torch.tensor([[0.20], [0.34], [0.57]])

# always use EVAL for fully trained models
model.eval()
model(new_inputs.to(device))
```
```python
tensor([[1.4185],
        [1.6908],
        [2.1381]], device='cuda:0', grad_fn=<AddmmBackward>)
```

### Recap 12. Realizing the importance of **setting the mode** of the model: **train()** or **eval()**, for **checkpointing** or **deploying** for prediction, respectively.

After loading the model, **DO NOT FORGET TO SET THE MODE**:
- **checkpointing**: model.train()
- **deploying / making predictions**: model.eval()

<hr style="border:2px solid blue">

***

## Chapter 2.1: Going Classy

We will develop a fully functioning class that implements all methods relevant to model training and evaluation. From now on, we'll use it over and over again to tackle different tasks and models.

```python
# StepByStep class

class StepByStep(object):
    def __init__(self, model, loss_fn, optimizer):
        # Here we define the attributes of our class
        
        # We start by storing the arguments as attributes 
        # to use them later
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Let's send the model to the specified device right away
        self.model.to(self.device)

        # These attributes are defined here, but since they are
        # not informed at the moment of creation, we keep them None
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        
        # These attributes are going to be computed internally
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

        # Creates the train_step function for our model, 
        # loss function and optimizer
        # Note: there are NO ARGS there! It makes use of the class
        # attributes directly
        self.train_step_fn = self._make_train_step_fn()
        # Creates the val_step function for our model and loss
        self.val_step_fn = self._make_val_step_fn()

    def to(self, device):
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (to be used later in
        # the mini-batches) and sends the model to the device
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        # This method allows the user to define which train_loader (and val_loader, optionally) to use
        # Both loaders are then assigned to attributes of the class
        # So they can be referred to later
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_tensorboard(self, name, folder='runs'):
        # This method allows the user to define a SummaryWriter to interface with TensorBoard
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def _make_train_step_fn(self):
        # This method does not need ARGS... it can refer to
        # the attributes: self.model, self.loss_fn and self.optimizer
        
        # Builds function that performs a step in the train loop
        def perform_train_step_fn(x, y):
            # Sets model to TRAIN mode
            self.model.train()

            # Step 1 - Computes our model's predicted output - forward pass
            yhat = self.model(x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)
            # Step 3 - Computes gradients for both "a" and "b" parameters
            loss.backward()
            # Step 4 - Updates parameters using gradients and the learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Returns the loss
            return loss.item()

        # Returns the function that will be called inside the train loop
        return perform_train_step_fn
    
    def _make_val_step_fn(self):
        # Builds function that performs a step in the validation loop
        def perform_val_step_fn(x, y):
            # Sets model to EVAL mode
            self.model.eval()

            # Step 1 - Computes our model's predicted output - forward pass
            yhat = self.model(x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)
            # There is no need to compute Steps 3 and 4, 
            # since we don't update parameters during evaluation
            return loss.item()

        return perform_val_step_fn
            
    def _mini_batch(self, validation=False):
        # The mini-batch can be used with both loaders
        # The argument `validation`defines which loader and 
        # corresponding step function is going to be used
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None
            
        # Once the data loader and step function, this is the 
        # same mini-batch loop we had before
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)
        return loss

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def train(self, n_epochs, seed=42):
        # To ensure reproducibility of the training process
        self.set_seed(seed)

        for epoch in range(n_epochs):
            # Keeps track of the numbers of epochs
            # by updating the corresponding attribute
            self.total_epochs += 1

            # inner loop
            # Performs training using mini-batches
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            # VALIDATION
            # no gradients in validation!
            with torch.no_grad():
                # Performs evaluation using mini-batches
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            # If a SummaryWriter has been set...
            if self.writer:
                scalars = {'training': loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                # Records both losses for each epoch under the main tag "loss"
                self.writer.add_scalars(main_tag='loss',
                                        tag_scalar_dict=scalars,
                                        global_step=epoch)

        if self.writer:
            # Closes the writer
            self.writer.close()

    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {'epoch': self.total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': self.losses,
                      'val_loss': self.val_losses}

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        # Loads dictionary
        checkpoint = torch.load(filename)

        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']

        self.model.train() # always use TRAIN for resuming training   

    def predict(self, x):
        # Set is to evaluation mode for predictions
        self.model.eval() 
        # Takes aNumpy input and make it a float tensor
        x_tensor = torch.as_tensor(x).float()
        # Send input to device and uses model for prediction
        y_hat_tensor = self.model(x_tensor.to(self.device))
        # Set it back to train mode
        self.model.train()
        # Detaches it, brings it to CPU and back to Numpy
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def add_graph(self):
        # Fetches a single mini-batch so we can use add_graph
        if self.train_loader and self.writer:
            x_sample, y_sample = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_sample.to(self.device))
```

```python
# data preparation

torch.manual_seed(13)

# Builds Tensor from Numpy array before split
x_tensor = torch.as_tensor(x).float()
y_tensor = torch.as_tensor(y).float()

# Build Dataset containing all datapoint
dataset = TensorDataset(x_tensor, y_tensor)

# Split Dataset
ratio = 0.8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train

train_data, val_data = random_split(dataset, [n_train, n_val])

# Build Data loader for training data and validation data
train_loader = DataLoader(
    dataset=train_data,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_data,
    batch_size=16
)
```

```python
# model configuration

# Sets learning rate
lr = 0.1

torch.manual_seed(42)
# Now we can create a model and send it at once to the device
model = nn.Sequential(nn.Linear(1, 1))

# Defines a SGD optimizer to update the parameters
# (now retrieved directly from the model)
optimizer = optim.SGD(model.parameters(), lr=lr)

# Defines a MSE loss function
loss_fn = nn.MSELoss(reduction='mean')
```

How to utilize the StepByStep class?
- Init StepByStep Object
```python
sbs = StepByStep(model, loss_ln, optimizer)
sbs.set_loaders(train_loader, val_loader)
sbs.set_tensorboard('classy')
```

- Train Model
```python
sbs.train(n_epochs=200)

print(model.state_dict())
print(sbs.total_epochs)
```
```python
OrderedDict([('0.weight', tensor([[1.9414]], device='cuda:0')), ('0.bias', tensor([1.0233], device='cuda:0'))])
200
```

- Plot Loss
```python
fig = sbs.plot_losses()
```
![alt text](md_images\image-5.png)

- Make Prediction
```python
new_data = np.array([.5, .3, .7]).reshape(-1, 1)
new_data
```
```python
array([[0.5],
       [0.3],
       [0.7]])
```
```python
predictions = sbs.predict(new_data)
predictions
```
```python
array([[1.9939734],
       [1.6056864],
       [2.3822603]], dtype=float32)
```

- Checkpointing
```python
sbs.save_checkpoin('model_checkpoint.pth')
```

- Resuming Training
```python
new_sbs = StepByStep(model, loss_fn, optimizer)
new_sbs.load_checkpoint('model_checkpoint.pth')
new_sbs.set_loaders(train_loader, val_loader)
new_sbs.train(n_epochs=50)
```
![alt text](md_images\image-6.png)

<hr style="border:2px solid blue">

***

## Chapter 3: A Simple Classification Problem
### Recap
1. Define a **binary classification problem**
2. Generating and preparing a toy dataset using Scikit-Learn's **make_moons()**
3. Defining **logits** as the result of a **linear combination of features**
4. Understanding what **odds ratios** and **log odds ratios** are
5. Figuring out we can **interpret logits as log odd ratios**
6. Mapping **logits into probabilities** using a **sigmoid function**
7. Defining a **logistic regression** as a **simple neural network with a sigmoid function in the output**
8. Understanding the **binary cross-entropy loss** and its PyTorch implementation **nn.BCELoss()**
9. Highlighting the **importance of choosing the correct combination of the last layer and loss function**
10. Using PyTorch's loss functions' arguments to handle **imbalanced dataset**
11. **Configuring** model, loss function, and optimizer for a classification problem
12. **Training** a model using the StepByStep class
13. Understanding that the validation loss **may be lower** than the training loss
14. **Making predictions** and mapping **predicted logits to probabilities**
15. **Using a classification threshold** to convert **probabilities into class**
16. Understanding the definition of a **decision boundary**
17. Understanding the concept of **separability of classes** and how it's related to **dimension**
18. Exploring **different classification thresholds** and their effect on the the **confusion matrix**
19. Reviewing typical **metrics** for evaluating classification algorithms, like true and false positive rates, precision, and recall
20. Building **ROS** and **precision-recall** curves out of **metric computed for multiple thresholds**
21. Understanding the reason behind the **quirk of losing precision** while raising the classification threshold
22. Defining the **best** and **worst** possible ROC and PR curves
23. Using the **area under the curve** to **compare different models**

### Recap 1: Define a **binary classification problem**

| Color | Value | Class |
| ----------- | ----------- | ----------- |
| Red | 0 | Negative |
| Blue | 1 | Positive |

![alt text](md_images\image-8.png)

![alt text](md_images\image-9.png)

![alt text](md_images\image-10.png)

![alt text](md_images\image-11.png)

![alt text](md_images\image-12.png)

![alt text](md_images\image-13.png)

![alt text](md_images\image-15.png)

![alt text](md_images\image-14.png)

![alt text](md_images\image-16.png)

![alt text](md_images\image-17.png)

![alt text](md_images\image-18.png)

### Recap 2: Generating and preparing a toy dataset using Scikit-Learn's **make_moons()**
```python
X, y = make_moons(
    n_samples=100,
    noise=0.3,
    random_state=0
)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=13
)

sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)
X_val = sc.transform(X_val)
```
![alt text](md_images\image-7.png)
***
# Computer Vision


***
# Sequences


***
# Natural Language Processing


***
# CHAPTER 1: GETTING STARTED WITH TIME SERIES
### Recap
1. **Loading** a time series using pandas
2. **Visualizing** a time series
3. **Resampling** a time series
4. **Dealing** with missing value
5. **Decomposing** a time series
6. Computing **autocorrelation**
7. Detecting **stationary**
8. Dealing with **heteroskedasticity**
9. **Loading and visualizing** a **multivariate** time series
10. **Resampling** a multivariate time series

### Recap 1. **Loading** a time series using pandas

```python
import pandas as pd

data = pd.read_csv('assets/datasets/time_series_solar.csv',
                   parse_dates=['Datetime'],
                   index_col='Datetime')

series = data['Incoming Solar']
```

```python
Datetime
2007-10-01 00:00:00    0.0
2007-10-01 01:00:00    0.0
2007-10-01 02:00:00    0.0
2007-10-01 03:00:00    0.0
2007-10-01 04:00:00    0.0
                      ... 
2013-09-30 19:00:00    0.0
2013-09-30 20:00:00    0.0
2013-09-30 21:00:00    0.0
2013-09-30 22:00:00    0.0
2013-09-30 23:00:00    0.0
Name: Incoming Solar, Length: 52608, dtype: float64
```

### Recap 2: **Visualizing** a time series

```python
series.plot(
    figsize=(12, 6),
    title='Solar radiation time series'
)

series_df = series.reset_index()

plt.rcParams['figure.figsize'] = [12, 6]

sns.set_theme(style='darkgrid')

sns.lineplot(
    data=series_df,
    x='Datetime',
    y='Incoming Solar'
)

plt.ylabel('Solar Radiation')
plt.xlabel('')
plt.title('Solar radition time series')
plt.show()
```

![alt text](md_images\image-19.png)

### Recap 3: **Resampling** a time series

Time series resampling is the process of changing the frequency of a time
series, for example, from hourly to daily. 

```python
series_daily = series.resample('D').sum()

series_df = series_daily.reset_index()

plt.rcParams['figure.figsize'] = [12, 6]

sns.set_theme(style='darkgrid')

sns.lineplot(x='Datetime',
             y='Incoming Solar',
             data=series_df)

plt.ylabel('Solar Radiation')
plt.xlabel('')
plt.title('Daily total solar radiation')
plt.show()
```
![alt text](md_images\image-21.png)

### Recap 4: **Dealing** with missing value

In datasets without temporal order, it is common to impute missing values using central statistics such as the mean or median

```python
sample_with_nan = series.head(365 * 2).copy()

size_na = int(0.6 * len(sample_with_nan))

idx = np.random.choice(a=range(len(sample_with_nan)),
                       size=size_na,
                       replace=False)

sample_with_nan[idx] = np.nan

# imputation with mean value
avg_value = sample_with_nan.mean()
imp_mean = sample_with_nan.fillna(avg_value)
# imputation with last known observation
imp_ffill = sample_with_nan.ffill()
# imputation with next known observation
imp_bfill = sample_with_nan.bfill()

plt.rcParams['figure.figsize'] = [12, 8]

fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, sharex=True)
# fig.suptitle('Time series imputation methods')

ax0.plot(sample_with_nan)
ax0.set_title('Original series with missing data')
ax1.plot(imp_mean)
ax1.set_title('Series with mean imputation')
ax2.plot(imp_ffill)
ax2.set_title('Series with ffill imputation')
ax3.plot(imp_bfill)
ax3.set_title('Series with bfill imputation')

plt.tight_layout()
```
![alt text](md_images\image-22.png)

### Recap 5: **Decomposing** a time series
Time series decomposition is the process of spliting a time series into its basic components, such as trend or seasonality