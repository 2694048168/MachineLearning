#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: chapter02.py
@Function: Binary Image Classification
@Python Version: 3.8
@Author: Wei Li
@Date：2021-07-09
"""


"""
# 1. Exploring the dataset
# 2. Creating a custom dataset
# 3. Splitting the dataset
# 4. Transforming the data
# 5. Creating dataloaders
# 6. Building the classification model
# 7. Defining the loss function
# 8. Defining the optimizer
# 9. Training and evaluation of the model
# 10. Deploying the model
# 11. Model inference on test data
"""

# -------------------
# Import packages
# -------------------
import os
import copy
# ------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.pylab as plt
from PIL import Image, ImageDraw
# ------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
# ------------------------------------------------
from torchsummary import summary
# ------------------------------------------------
import torchvision
from torchvision import transforms


# -------------------------
# 保证随机状态一致
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True #为了提升计算速度
torch.backends.cudnn.benchmark = False # 避免因为随机性产生出差异
np.random.seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -------------------------
# Exploring the dataset
# -------------------------
# https://www.kaggle.com/c/histopathologic-cancer-detection/data

path2image = r"./image"
path2csv = r"./cancer_data/train_labels.csv"
labels_df = pd.read_csv(path2csv)
print(labels_df.head())

print(labels_df['label'].value_counts())
labels_df["label"].hist()
plt.savefig(os.path.join(path2image, "2.png"))
# plt.show()
plt.close()

path2train = r"./cancer_data/train"
color = False

malignantIds = labels_df.loc[labels_df["label"]==1]["id"].values

plt.rcParams["figure.figsize"] = (10.0, 10.0)
plt.subplots_adjust(wspace=0, hspace=0)
nrows, ncols = 3, 3
for i, id_ in enumerate(malignantIds[:nrows*ncols]):
    full_filenames = os.path.join(path2train, id_ + ".tif")

    img = Image.open(full_filenames)

    # draw 32 * 32 rectangle
    draw = ImageDraw.Draw(img)
    draw.rectangle(((32,32), (64,64)), outline="green")

    plt.subplot(nrows, ncols, i+1)
    if color is True:
        plt.show(np.array(img))
    else:
        plt.imshow(np.array(img)[:,:,0], cmap="gray")
    
    plt.axis("off")
    
plt.savefig(os.path.join(path2image, "3.png"))
# plt.show()
plt.close()
print(f"Image shape: {np.array(img).shape}")
print(f"The pixel values range from {np.min(img)} to {np.max(img)}")


# -------------------------
# Create a Custom Dataset
# -------------------------
class histoCancerDataset(Dataset):
    def __init__(self, data_dir, transform, data_type="train"):
        super(histoCancerDataset, self).__init__()

        # image data
        path2data = os.path.join(data_dir, data_type)
        filenames = os.listdir(path2data)
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]

        # image label
        path2csvLabels = os.path.join(data_dir, "train_labels.csv")
        labels_df = pd.read_csv(path2csvLabels)
        # set data frame index to id
        labels_df.set_index("id", inplace=True)
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in filenames]

        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self, index):
        image = Image.open(self.full_filenames[index])
        image = self.transform(image)
        return image, self.labels[index]

data_transformer = transforms.Compose([transforms.ToTensor()])
data_dir = r"./cancer_data/"
histo_dataset = histoCancerDataset(data_dir=data_dir, transform=data_transformer, data_type="train")
print(len(histo_dataset))
img, lable = histo_dataset[9]
print(img.shape, torch.min(img), torch.max(img))


# -------------------------
# Split Dataset
# -------------------------
len_histo = len(histo_dataset)
len_train = int(0.8 * len_histo)
len_val = len_histo - len_train

train_ds, val_ds = random_split(histo_dataset, [len_train, len_val])
print(f"The train dataset length is : {len(train_ds)}")
print(f"The validation dataset length is : {len(val_ds)}")

for x, y in train_ds:
    print(f"The shape of train image is : {x.shape}")
    print(f"The shape of train lable is : {y}")
    break

for x, y in val_ds:
    print(f"The shape of valiation image is : {x.shape}")
    print(f"The shape of valiation lable is : {y}")
    break

def show(img, y, save_file, color=False):
    ndarray_img = img.numpy()
    # [CHW] --> [CHW]
    ndarray_img_tr = np.transpose(ndarray_img, (1,2,0))

    if color==False:
        ndarray_img_tr = ndarray_img_tr[:,:,0]
        plt.imshow(ndarray_img_tr, interpolation="nearest", cmap="gray")
    else:
        plt.imshow(ndarray_img_tr, interpolation="nearest")
    
    plt.title(f"lable: {str(y)}")
    plt.savefig(os.path.join(path2image, save_file))
    # plt.show()
    plt.close()

grid_size = 4
rnd_inds = np.random.randint(0, len(train_ds), grid_size)
print(f"The iamge indices: {rnd_inds}")

x_grid_train = [train_ds[i][0] for i in rnd_inds]
y_grid_train = [train_ds[i][1] for i in rnd_inds]
x_grid_train = torchvision.utils.make_grid(x_grid_train, nrows=4, padding=2)
print(x_grid_train.shape)

plt.rcParams["figure.figsize"] = (10.0, 5)
show(x_grid_train, y_grid_train, save_file="4.png")


grid_size = 4
rnd_inds = np.random.randint(0, len(val_ds), grid_size)
print(f"The iamge indices: {rnd_inds}")

x_grid_train = [val_ds[i][0] for i in rnd_inds]
y_grid_train = [val_ds[i][1] for i in rnd_inds]
x_grid_train = torchvision.utils.make_grid(x_grid_train, nrows=4, padding=2)
print(x_grid_train.shape)

plt.rcParams["figure.figsize"] = (10.0, 5)
show(x_grid_train, y_grid_train, save_file="5.png")


# -------------------------
# Transform Data
# -------------------------
train_transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(45),
    transforms.RandomResizedCrop(96, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    transforms.ToTensor(),
])

val_transformer = transforms.Compose([transforms.ToTensor()])

# overwrite the transform functions
train_ds.transform = train_transformer
val_ds.transform = val_transformer


# -------------------------
# Create Dataloader
# -------------------------
train_ds_batch = DataLoader(train_ds, batch_size=32, shuffle=True)
val_ds_batch = DataLoader(val_ds, batch_size=64, shuffle=False)

# extract batch from train data
for x, y in train_ds_batch:
    print(x.shape)
    print(y.shape)
    break

for x, y in val_ds_batch:
    print(x.shape)
    print(y.shape)
    break


# -----------------------------------
# Building Classification Model
# -----------------------------------
# dumb baselines
# -----------------------------------
y_val = [y for _, y in val_ds]

def accuracy(labels, out):
    return np.sum(out==labels) / float(len(labels))

acc_all_zeros = accuracy(y_val, np.zeros_like(y_val))

acc_all_ones = accuracy(y_val, np.ones_like(y_val))

acc_random = accuracy(y_val, np.random.randint(2, size=len(y_val)))

print(f"The Accuracy random prediction: {acc_random:.4f}")
print(f"The Accuracy all zero prediction: {acc_all_zeros:.4f}")
print(f"The Accuracy all one prediction: {acc_all_ones:.4f}")


# -----------------------------------
# find Output size
# -----------------------------------
def findConv2dOutShape(H_in, W_in, conv, pool=2):
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    H_out = np.floor((H_in+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    W_out = np.floor((W_in+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)

    if pool:
        H_out /= pool
        W_out /= pool

    return int(H_out), int(W_out)

conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0, dilation=1)
h, w = findConv2dOutShape(96, 96, conv1)
print(h, w)


# -------------------------
# Define Model
# -------------------------
class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()

        C_in, H_in, W_in = params["input_shape"]
        init_f = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]

        self.conv1 = nn.Conv2d(C_in, init_f, kernel_size=3)
        h, w = findConv2dOutShape(H_in, W_in, self.conv1)

        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv2)

        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h, w = findConv2dOutShape(h,w,self.conv3)

        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h, w = findConv2dOutShape(h,w,self.conv4)

        # compute the flatten size
        self.num_flatten = h * w * init_f * 8

        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, self.num_flatten)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout_rate)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

# dict to define model parameters
# json ???
params_model={
        "input_shape": (3,96,96),
        "initial_filters": 8,    
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 2,
        }

# create model
cnn_model = Net(params_model)

# move model to cuda/gpu device
cnn_model = cnn_model.to(device=device)

print(cnn_model)
print(next(cnn_model.parameters()).device)
summary(cnn_model, input_size=(3, 96, 96))


# -------------------------
# Loss function
# -------------------------
loss_func = nn.NLLLoss(reduction="sum")

n, c = 8, 2
y = torch.randn(n, c, requires_grad=True)
ls_F = nn.LogSoftmax(dim=1)
y_out = ls_F(y)
print(y_out.shape)

target = torch.randint(c, size=(n,))
print(target.shape)

loss = loss_func(y_out, target)
print(loss.item())

loss.backward()
print(y.data)


# -------------------------
# Defining Optimizer
# -------------------------
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group["lr"]


current_lr = get_lr(opt)
print(f"The current learning rate: {current_lr}")

# define learning rate scheduler
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20, verbose=1)

for i in range(100):
    lr_scheduler.step(1)


# -------------------------
# Training and Evaluation
# -------------------------
def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

n, c = 8, 2
output = torch.randn(n, c, requires_grad=True)
print (output)
print(output.shape)

#target = torch.randint(c,size=(n,))
target = torch.ones(n, dtype=torch.long)
print(target.shape)

print(metrics_batch(output,target))


def loss_batch(loss_func, output, target, opt=None):
    # get loss 
    loss = loss_func(output, target)
    # get performance metric
    metric_b = metrics_batch(output,target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        # move batch to device
        xb = xb.to(device)
        yb = yb.to(device)
        
        # get model output
        output = model(xb)
        # get loss per batch
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        # update running loss
        running_loss += loss_b
        # update running metric
        if metric_b is not None:
            running_metric += metric_b

        # break the loop in case of sanity check
        if sanity_check is True:
            break
    
    # average loss value
    loss = running_loss / float(len_data)
    
    # average metric value
    metric = running_metric / float(len_data)
    
    return loss, metric


def train_val(model, params):
    # extract model parameters
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]

    # history of loss values in each epoch
    loss_history={
        "train": [],
        "val": [],
    }
    
    # histroy of metric values in each epoch
    metric_history={
        "train": [],
        "val": [],
    }
    
    # a deep copy of weights for the best performing model
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # initialize best loss to a large value
    best_loss = float('inf')

    # main loop
    for epoch in range(num_epochs):
        
        # get current learning rate
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        
        # train model on training dataset
        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,sanity_check,opt)

        # collect loss and metric for training dataset
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
        # evaluate model on validation dataset    
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        
       
        # store best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
            # store weights into a local file
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
        
        # collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        # learning rate schedule
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts) 

        print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f" %(train_loss,val_loss,100*val_metric))
        print("-"*10) 

    # load best model weights
    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history


loss_func = nn.NLLLoss(reduction="sum")
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)

path2models = r"./models/cancer/"
os.makedirs(path2models, exist_ok=True)
savemodel2file = os.path.join(path2models, "weight.pt")
params_train={
 "num_epochs": 100,
 "optimizer": opt,
 "loss_func": loss_func,
 "train_dl": train_ds_batch,
 "val_dl": val_ds_batch,
 "sanity_check": True,
 "lr_scheduler": lr_scheduler,
 "path2weights": savemodel2file,
}

# train and validate the model
cnn_model, loss_hist, metric_hist = train_val(cnn_model, params_train)


# -------------------------
# Train-Validation Progress
# -------------------------
num_epochs = params_train["num_epochs"]

# plot loss progress
plt.title("Train-Val Loss")
plt.plot(range(1, num_epochs+1), loss_hist["train"], label="train")
plt.plot(range(1, num_epochs+1), loss_hist["val"], label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.savefig(os.path.join(path2image, "5.png"))
# plt.show()
plt.close()

# plot accuracy progress
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1), metric_hist["train"], label="train")
plt.plot(range(1,num_epochs+1), metric_hist["val"], label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.savefig(os.path.join(path2image, "6.png"))
# plt.show()
plt.close()
# ----------------------------------------------------------------------------------

loss_func = nn.NLLLoss(reduction="sum")
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)

params_train={
 "num_epochs": 2,
 "optimizer": opt,
 "loss_func": loss_func,
 "train_dl": train_ds_batch,
 "val_dl": val_ds_batch,
 "sanity_check": False,
 "lr_scheduler": lr_scheduler,
 "path2weights": savemodel2file,
}

# train and validate the model
cnn_model,loss_hist,metric_hist = train_val(cnn_model,params_train)

# Train-Validation Progress
num_epochs=params_train["num_epochs"]

# plot loss progress
plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.savefig(os.path.join(path2image, "7.png"))
# plt.show()
plt.close()

# plot accuracy progress
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.savefig(os.path.join(path2image, "8.png"))
# plt.show()
plt.close()
# ----------------------------------------------------------------------------------