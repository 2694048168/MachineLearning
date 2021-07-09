#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: chapter01.py
@Function: Getting Started with PyTorch for Deep Learning
@Python Version: 3.8
@Author: Wei Li
@Date：2021-07-08
"""


"""
# 1. Installing software tools and packages
# 2. Working with PyTorch tensors
# 3. Loading and processing data
# 4. Building models
# 5. Defining the loss function and optimizer
# 6. Training and evaluation
"""

# -------------------
# Import packages
# -------------------
import os
# ------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
# ------------------------------------------------
from torchsummary import summary
# ------------------------------------------------
import torchvision
from torchvision import datasets
from torchvision import utils
from torchvision import transforms


# --------------------------------
# verify installation
# --------------------------------
print("-----------------------------------------------")
print(f"The version of PyTorch is : {torch.__version__}")
print(f"The version of PyTorch for Vision is : {torchvision.__version__}")
print(f"The GPU or CUDA is available: {torch.cuda.is_available()}")
print(f"The number of GPU is: {torch.cuda.device_count()}")
print(f"The ID of GPU is: {torch.cuda.current_device()}")
print(f"The Name of GPU is: {torch.cuda.get_device_name(0)}")


# --------------------------------
# Tensor Data type
# --------------------------------
print("-----------------------------------------------")
x_tensor = torch.ones(2, 3, dtype=torch.float32)
print(x_tensor)
print(f"The type of tensor data is : {x_tensor.dtype}")


# ------------------------------------
# Converting Tensors to NumPy arrays
# ------------------------------------
print("-----------------------------------------------")
x = torch.rand(2, 3)
print(x)
print(f"The type of tensor data is : {x.dtype}")
y = x.numpy()
print(y)
print(f"The type of ndarray data is : {y.dtype}")


# ------------------------------------
# Converting NumPy arrays to Tensors
# ------------------------------------
print("-----------------------------------------------")
x_ndarray = np.zeros((2, 3), dtype=np.float32)
print(x_ndarray)
print(f"The type of ndarray data is : {x_ndarray.dtype}")
y_tensor = torch.from_numpy(x_ndarray)
print(y_tensor)
print(f"The type of tensor data is : {y_tensor.dtype}")


# ------------------------------------
# Move Tensors between devices
# ------------------------------------
print("-----------------------------------------------")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x_tensor_device = torch.tensor([1.5, 2])
print(x_tensor_device.device)
x_tensor_device.to(device)
x_tensor_cpu = torch.tensor([23, 34], device=device)
print(x_tensor_cpu.device)


# ------------------------------------
# Loading and Processing Data
# ------------------------------------
print("-----------------------------------------------")
path2data = r"./datasets"
os.makedirs(path2data, exist_ok=True)

train_data = datasets.MNIST(root=path2data, train=True, download=True)
val_data = datasets.MNIST(root=path2data, train=False, download=True)
x_train, y_train = train_data.data, train_data.targets
x_val, y_val = val_data.data, val_data.targets

print(f"The shape of train data: {x_train.shape}")
print(f"The shape of train target: {y_train.shape}")
print(f"The shape of val data: {x_val.shape}")
print(f"The shape of val target: {y_val.shape}")


# ------------------------
# Display images
# ------------------------
print("-----------------------------------------------")
# [CHW] --> [BCHW]
if len(x_train.shape) == 3:
    # tensor method
    x_train = x_train.unsqueeze(1)
    # x_train = torch.reshape(x_train, [-1, 1, 28, 28])
print(f"The train data shape : {x_train.shape}")

if len(x_val.shape) == 3:
    # tensor method
    x_val = x_val.unsqueeze(1)
print(f"The val data shape : {x_val.shape}")

x_grid = utils.make_grid(x_train[:40], nrow=8, padding=2)
print(f"The shape of grid based pytorch utils: {x_grid.shape}")

def show(img, path2image):
    ndarray_img = img.numpy()
    # [CHW] --> [HEC]
    ndarray_img = np.transpose(ndarray_img, (1, 2, 0))
    plt.imshow(ndarray_img, interpolation="nearest")
    plt.savefig(os.path.join(path2image, "0.png"))
    # plt.show()
    plt.close()

path2image = r"./image"
os.makedirs(path2image, exist_ok=True)
show(x_grid, path2image)


# ------------------------
# Transform Data
# ------------------------
print("-----------------------------------------------")
# train_data=datasets.MNIST(path2data, train=True, download=True, transform=data_transform)
data_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1),
                                     transforms.RandomVerticalFlip(p=1),
                                     transforms.ToTensor(),
                                    ])

img = train_data[0][0]
img_trans = data_transform(img)
img_trans_ndarray = img_trans.numpy()

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.subplot(1,2,2)
plt.imshow(img_trans_ndarray[0], cmap="gray")
plt.title("Transform Image")
plt.savefig(os.path.join(path2image, "1.png"))
# plt.show()
plt.close()


# ---------------------------
# Wrap Tensors into Dataset
# ---------------------------
print("-----------------------------------------------")
train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)

for x, y in train_ds:
    print(x.shape, y.item())
    break

for x, y in val_ds:
    print(x.shape, y.item())
    break


# ---------------------------
# Iterate Over Dataset
# ---------------------------
print("-----------------------------------------------")
train_ds_batch = DataLoader(train_ds, batch_size=32)
val_ds_batch = DataLoader(val_ds, batch_size=32)

for x_batch, y_batch in train_ds_batch:
    print(x_batch.shape)
    print(y_batch.shape)
    break

for x_batch, y_batch in val_ds_batch:
    print(x_batch.shape)
    print(y_batch.shape)
    break


# ---------------------------
# Building Models
# ---------------------------
print("-----------------------------------------------")
input_tensor = torch.randn(64,  1000)
linear_layer = nn.Linear(1000, 100)
output = linear_layer(input_tensor)
print(output.size())


# -----------------------------------
# Define models using nn.Sequential
# -----------------------------------
print("-----------------------------------------------")
model = nn.Sequential(nn.Linear(4, 5),
                      nn.ReLU(),
                      nn.Linear(5, 1))
print(model)


# -----------------------------------
# Define models using nn.Module
# -----------------------------------
print("-----------------------------------------------")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, 
                               kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, 
                               kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(4*4*16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

model = Net()
print(model)


# -----------------------------------
# Move Model to Device
# -----------------------------------
print("-----------------------------------------------")
print(next(model.parameters()).device)
model.to(device)
print(next(model.parameters()).device)


# -----------------------------------
# Show model summary
# -----------------------------------
print("-----------------------------------------------")
summary(model, input_size=(1, 28, 28))


# -----------------------------------
# Loss Function
# -----------------------------------
loss_function = nn.NLLLoss(reduction="sum")

for x_batch, y_batch in train_ds_batch:
    x_batch = x_batch.type(torch.float32).to(device)
    y_batch = y_batch.to(device)

    out = model(x_batch)

    loss = loss_function(out, y_batch)
    print(loss.item())
    break

# compute gradients
loss.backward()


# -----------------------------------
# Optimizer
# -----------------------------------
opt = optim.Adam(model.parameters(), lr=1e-4)

# update model paras
opt.step()

# set gradients to zero
opt.zero_grad()


# -----------------------------------
# Training and Validation
# -----------------------------------
print("-----------------------------------------------")
def metrics_batch(target, output):
    # obtain output class
    pred = output.argmax(dim=1, keepdim=True)
    
    # compare output class with target class
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


def loss_batch(loss_func, xb, yb, yb_h, opt=None):
    
    # obtain loss
    loss = loss_func(yb_h, yb)
    
    # obtain performance metric
    metric_b = metrics_batch(yb,yb_h)
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), metric_b


def loss_epoch(model, loss_func, dataset_dl, opt=None):
    loss = 0.0
    metric = 0.0
    len_data = len(dataset_dl.dataset)
    for xb, yb in dataset_dl:
        xb = xb.type(torch.float).to(device)
        yb = yb.to(device)
        
        # obtain model output
        yb_h = model(xb)

        loss_b, metric_b = loss_batch(loss_func, xb, yb,yb_h, opt)
        loss += loss_b
        if metric_b is not None:
            metric += metric_b
    loss /= len_data
    metric /= len_data
    return loss, metric


def train_val(epochs, model, loss_func, opt, train_dl, val_dl):
    for epoch in range(epochs):
        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,opt)
        
            
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl)
        
        accuracy=100*val_metric

        print("epoch: %d, train loss: %.6f, val loss: %.6f, accuracy: %.2f" %(epoch, train_loss,val_loss,accuracy))


epochs = 5
train_val(epochs, model, loss_function, opt, train_ds_batch, val_ds_batch)


# -----------------------------------
# Store and Load Models
# -----------------------------------
path2models = r"./models"
os.makedirs(path2models, exist_ok=True)
filename_model = os.path.join(path2models, "mnist.pth")
torch.save(model, filename_model)

_model = Net()
_model = torch.load(filename_model)
_model.eval()
_model.to(device)

# -----------------------------------
# Deploy Models
# -----------------------------------
# x is a data point with C*H*W shape
n = 100
x = x_val[n]
y = y_val[n]
print(x.shape)
plt.imshow(x.numpy()[0],cmap="gray")
plt.show()
plt.close()

# we use unsqueeze to expand dimensions to 1*C*H*W
x = x.unsqueeze(0)

# convert to torch.float32
x = x.type(torch.float)

# move to cuda device
x = x.to(device)

# get model output
output = _model(x)

# get predicted class
pred = output.argmax(dim=1, keepdim=True)
print (pred.item(),y.item())