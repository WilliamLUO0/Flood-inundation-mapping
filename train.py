# Rapid mapping of flood inundation by deep learning-based image super-resolution
# Developer: Wenke Song
# The University of Hong Kong
# Contact email: songwk@connect.hku.hk
# MIT License
# Copyright (c) 2024 songwk0924

from DenseUnet import *
from ResUnet import *
from Unet import *
from dataloader import *
import torch
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import os
import time
import matplotlib.pyplot as plt
# import wandb

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="DenseUnet-project",
    
#     # track hyperparameters and run metadata
#     config={
#     "Dynamic learning_rate": 0.001,
#     "architecture": "DenseUnet",
#     "dataset": "FloodSimulation",
#     "epochs": 500,
#     }
# )
# os.environ["NCCL_DEBUG"] = "INFO"
seed = 42
date = f"DenseUnet/minseed{seed}new"  # 1MAE 2-15, 20h~1235
# load data
bs = 256
# random split
dataset = NpyDataset('/media/hd02/wksong/dm/pftmin', '/media/hd02/wksong/dm/lftmin')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(seed) 
train_dataset, val_dataset = random_split(dataset, [train_size, val_size],generator=generator)

train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True,num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=True,num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model 
model = DenseUnet(in_channels=5, growth_rate=32)
# model = ResUnet(in_channels=5, growth_rate=32)
# model = ResUnet(in_channels=5, growth_rate=32)
model.float()

if torch.cuda.device_count() > 1:
    print("Use ", torch.cuda.device_count(), " GPUs")
    model = nn.DataParallel(model)

model = model.to(device)


# loss function

# Then replace the original loss function with the new one
def masked_loss(outputs, labels, mask):
    # Apply mask to outputs and labels
    mask = mask.unsqueeze(1)

    masked_outputs = outputs * mask
    masked_labels = labels * mask

    # MSE
    # loss = torch.sum((masked_outputs - masked_labels) ** 2) / torch.sum(mask)
    # MAE
    loss = torch.sum(torch.abs(masked_outputs - masked_labels)) / torch.sum(mask)
    
    return loss

# optim
learning_rate = 1*1e-3
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
# set training parameters
total_train_step = 0
total_test_step = 0
epochs = 500

start_time = time.time()
kk = 1
best_loss = 1000

# train
for epoch in range(epochs):
    print("----Training epoch: [{}]----".format(epoch+1))
    for param_group in optimizer.param_groups:
        print("Current learning rate is: {}".format(param_group['lr']))
    # train step
    model.train()
    total_train_loss = 0
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()
        
        # mask
        mask = inputs[:, 0, :, :]
        mask = (mask != 0).float().to(device)

        # forward
        outputs = model(inputs,mask)
        loss = masked_loss(outputs,labels, mask)
        total_train_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print
        total_train_step = total_train_step + 1

        # wandb.log({"Train loss": loss})

        if total_train_step % 30 == 0:
            end_time = time.time()
            print("Train step: ({}), Loss: {:.6f}, time: {:.2f}s".format(total_train_step, loss.item(),end_time-start_time))
    avg_train_loss = total_train_loss / len(train_dataloader)
    
    print("Epoch [{}/{}], Avg train Loss: {:.6f}".format(epoch+1, epochs, avg_train_loss))
    # wandb.log({"Avg_train loss": avg_train_loss},step=epoch)
    
    # val step
    model.eval()
    total_test_loss = 0
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_dataloader):
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()

            # mask
            mask = inputs[:, 0, :, :]
            mask = (mask != 0).float().to(device)

            # forward
            outputs = model(inputs, mask)
            loss = masked_loss(outputs,labels, mask)
            #loss = loss_fn(outputs, labels)

            total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(val_dataloader)
        # losss = avg_test_loss/2 + avg_train_loss/2
        scheduler.step(avg_test_loss)

        # Save the model if it has the lowest avg_test_loss so far
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model, "/outputpthd/{}/model_best.pth".format(date))
            print("model_best.pth have saved")

        # wandb.log({"Avg_train loss": avg_train_loss,"Avg_test loss": avg_test_loss},step=epoch)
        # wandb.log({"Accuracy": accuracy},step=epoch)
    
    print("Epoch [{}/{}], Avg test Loss: {:.6f}".format(epoch+1, epochs, avg_test_loss))

    if (epoch+1) % 50 == 0:
        torch.save(model, "/outputpthd/{}/model_{}.pth".format(date,epoch+1))
        print("model_{}.pth have saved".format(epoch+1))

torch.save(model, "/outputpthd/{}/model_final.pth".format(date))
print("model_final.pth have saved")
# wandb.finish()

