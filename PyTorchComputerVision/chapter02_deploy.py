#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: chapter02_deploy.py
@Function: Binary Image Classification
@Python Version: 3.8
@Author: Wei Li
@Date：2021-07-09
"""


"""
# 10. Deploying the model
# 11. Model inference on test data
"""

# -------------------
# Import packages
# -------------------
import os
import time
# ------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score
# ------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import random_split
# ------------------------------------------------
from torchvision import utils
import torchvision.transforms as transforms



# -------------------------
# 保证随机状态一致
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True #为了提升计算速度
torch.backends.cudnn.benchmark = False # 避免因为随机性产生出差异
np.random.seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------
def findConv2dOutShape(H_in,W_in,conv,pool=2):
    # get conv arguments
    kernel_size=conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    # Ref: https://pytorch.org/docs/stable/nn.html
    H_out=np.floor((H_in+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    W_out=np.floor((W_in+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        H_out/=pool
        W_out/=pool
    return int(H_out),int(W_out)

# ----------------------
class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
    
        C_in,H_in,W_in=params["input_shape"]
        init_f=params["initial_filters"] 
        num_fc1=params["num_fc1"]  
        num_classes=params["num_classes"] 
        self.dropout_rate=params["dropout_rate"] 
        
        self.conv1 = nn.Conv2d(C_in, init_f, kernel_size=3)
        h,w=findConv2dOutShape(H_in,W_in,self.conv1)
        
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv2)
        
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv3)

        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv4)
        
        # compute the flatten size
        self.num_flatten=h*w*8*init_f
        
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
        x=F.dropout(x, self.dropout_rate)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ------------------------
# model parameters
params_model={
 "input_shape": (3,96,96),
 "initial_filters": 8, 
 "num_fc1": 100,
 "dropout_rate": 0.25,
 "num_classes": 2,
 }

# initialize model
cnn_model = Net(params_model)


# --------------------------
path2weights = r"./models/cancer/weight.pt"
# load state_dict into model
cnn_model.load_state_dict(torch.load(path2weights))

# set model in evaluation mode
cnn_model.eval()
cnn_model=cnn_model.to(device)

# ------------------------------------------------------------------------
def deploy_model(model,dataset,device, num_classes=2,sanity_check=False):

    len_data=len(dataset)
    
    # initialize output tensor on CPU: due to GPU memory limits
    y_out=torch.zeros(len_data,num_classes)
    
    # initialize ground truth on CPU: due to GPU memory limits
    y_gt=np.zeros((len_data),dtype="uint8")
    
    # move model to device
    model=model.to(device)
    
    elapsed_times=[]
    with torch.no_grad():
        for i in range(len_data):
            x,y=dataset[i]
            y_gt[i]=y
            start=time.time()    
            y_out[i]=model(x.unsqueeze(0).to(device))
            elapsed=time.time()-start
            elapsed_times.append(elapsed)

            if sanity_check is True:
                break

    inference_time=np.mean(elapsed_times)*1000
    print("average inference time per image on %s: %.2f ms " %(device,inference_time))
    return y_out.numpy(),y_gt


# ----------------------------------------------
class histoCancerDataset(Dataset):
    def __init__(self, data_dir, transform,data_type="train"):      
    
        # path to images
        path2data=os.path.join(data_dir,data_type)

        # get a list of images
        self.filenames = os.listdir(path2data)

        # get the full path to images
        self.full_filenames = [os.path.join(path2data, f) for f in self.filenames]

        # labels are in a csv file named train_labels.csv
        csv_filename=data_type+"_labels.csv"
        path2csvLabels=os.path.join(data_dir,csv_filename)
        labels_df=pd.read_csv(path2csvLabels)

        # set data frame index to id
        labels_df.set_index("id", inplace=True)

        # obtain labels from data frame
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in self.filenames]

        self.transform = transform
      
    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)
      
    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]

# ----------------------------------------------
data_transformer = transforms.Compose([transforms.ToTensor()])

data_dir = r"./cancer_data"
histo_dataset = histoCancerDataset(data_dir, data_transformer, "train")
print(len(histo_dataset))

len_histo=len(histo_dataset)
len_train=int(0.8*len_histo)
len_val=len_histo-len_train

train_ds,val_ds=random_split(histo_dataset,[len_train,len_val])

print("train dataset length:", len(train_ds))
print("validation dataset length:", len(val_ds))

# deploy model 
y_out, y_gt = deploy_model(cnn_model,val_ds,device=device,sanity_check=False)
print(y_out.shape,y_gt.shape)


# -----------------
# Accuracy
# -----------------
# get predictions
y_pred = np.argmax(y_out,axis=1)
print(y_pred.shape,y_gt.shape)

# compute accuracy 
acc=accuracy_score(y_pred,y_gt)
print("accuracy: %.4f" %acc)


# -----------------
# Deploy on CPU
# -----------------
device_cpu = torch.device("cpu")
y_out,y_gt=deploy_model(cnn_model,val_ds,device=device_cpu,sanity_check=False)
print(y_out.shape,y_gt.shape)


# --------------------------------
# Model Inference on Test Data
# --------------------------------
path2csv = r"./cancer_data/test_labels.csv"
labels_df = pd.read_csv(path2csv)
print(labels_df.head())

data_dir = r"./cancer_data"
histo_test = histoCancerDataset(data_dir, data_transformer,data_type="test")
print(len(histo_test))

y_test_out,_=deploy_model(cnn_model,histo_test, device, sanity_check=False)
y_test_pred=np.argmax(y_test_out,axis=1)
print(y_test_pred.shape)


# ----------------------------
path2image = r"./image"
def show(img,y,path2image, color=False):
    # convert tensor to numpy array
    npimg = img.numpy()
   
    # Convert to H*W*C shape
    npimg_tr=np.transpose(npimg, (1,2,0))
    
    if color==False:
        npimg_tr=npimg_tr[:,:,0]
        plt.imshow(npimg_tr,interpolation='nearest',cmap="gray")
    else:
        # display images
        plt.imshow(npimg_tr,interpolation='nearest')
    plt.title("label: "+str(y))

    plt.savefig(os.path.join(path2image, "9.png"))
    plt.close()
    
grid_size=4
rnd_inds=np.random.randint(0,len(histo_test),grid_size)
print("image indices:",rnd_inds)

x_grid_test=[histo_test[i][0] for i in range(grid_size)]
y_grid_test=[y_test_pred[i] for i in range(grid_size)]

x_grid_test=utils.make_grid(x_grid_test, nrow=4, padding=2)
print(x_grid_test.shape)

plt.rcParams['figure.figsize'] = (10.0, 5)
show(x_grid_test,y_grid_test, path2image)


# --------------------------------
# Create Submission
# --------------------------------
print(y_test_out.shape)
cancer_preds = np.exp(y_test_out[:, 1])
print(cancer_preds.shape)

# get test id's from the sample_submission.csv 
path2sampleSub = "./cancer_data/sample_submission.csv"

# read sample submission
sample_df = pd.read_csv(path2sampleSub)

# get id column
ids_list = list(sample_df.id)

# convert predictions to to list
pred_list = [p for p in cancer_preds]

# create a dict of id and prediction
pred_dic = dict((key[:-4], value) for (key, value) in zip(histo_test.filenames, pred_list))    


# re-order predictions to match sample submission csv 
pred_list_sub = [pred_dic[id_] for id_ in ids_list]

# create convert to data frame
submission_df = pd.DataFrame({'id':ids_list,'label':pred_list_sub})

# Export to csv
submission = r"./cancer_data/subminssions"
if not os.path.exists(submission):
    os.makedirs(submission)
    print("submission folder created!")
    
path2submission = os.path.join(submission, "submission.csv")
submission_df.to_csv(path2submission, header=True, index=False)
print(submission_df.head())