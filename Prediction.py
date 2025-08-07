# Rapid mapping of flood inundation by deep learning-based image super-resolution
# Developer: Wenke Song
# The University of Hong Kong
# Contact email: songwk@connect.hku.hk
# MIT License
# Copyright (c) 2024 songwk0924

import torch
import numpy as np
import pandas as pd
from DenseResUnetnewsimplemaxprelu import *
import time

start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prediction
floodindex = 'v' # d -- max water depth, v -- max velocity
coarsegrid = 80
rainfallevent = 'r2'
arch = 'DenseUnet'

model = torch.load(f'./Trained_model/model_{floodindex}_{arch}.pth',map_location=device)

nsize = 128
overlap_rate = 0.5
vminth = 0.1

# Test cases
data = np.load(f'./Test_cases/Test_{floodindex}_{rainfallevent}.npy')

dataold = np.copy(data)
bathy_patch = data[:, 0, :, :]
mask_patch = (bathy_patch != 0)

# Normalize data
channels_to_normalize = [0, 1, 2, 3, 4]
for c in channels_to_normalize:
    min_val = data[:, c, ...].min(axis=(0, 1, 2), keepdims=True)
    max_val = data[:, c, ...].max(axis=(0, 1, 2), keepdims=True)

    if c == 2:
        data[:, c, ...] = (data[:, c, ...]) / 6
    elif c == channels_to_normalize[-1]:
        min_val1 = np.array([[[0.]]])

        if floodindex == 'd':
            max_val1 = np.array([[[6.17581013]]]) # max water depth in the training data 
        
        elif floodindex == 'v':
            max_val1 = np.array([[[6.89075068]]]) # max velocity in the training data

        data[:, c, ...] = (data[:, c, ...] - min_val1) / (max_val1 - min_val1)
    
    else:
        data[:, c, ...] = (data[:, c, ...] - min_val) / (max_val - min_val)


data = torch.from_numpy(data) # {features}
data = data.to(device).float()
mask_patch = torch.from_numpy(mask_patch).to(device).float()

# Load fine gird flood maps
if floodindex == 'd':
    finedata_path = f'./Test_cases/Fine_grid_flood_maps/hmax_{rainfallevent}.asc' 
elif floodindex == 'v':
    finedata_path = f'./Test_cases/Fine_grid_flood_maps/velmax_{rainfallevent}.asc'


finedata_d = pd.read_csv(finedata_path,header=None)

bathy_5m = np.array(pd.read_csv('./Test_cases/bathy_mat_5m_0p.csv', delimiter=' ',header=None))
mask = np.array((bathy_5m != 0)).astype(int)

fd = []
for i in range(6, finedata_d.shape[0]):
    row_d_values2 = finedata_d.iloc[i].values[0].split()
    row_d_values2 = np.array(list(map(float, row_d_values2)))
    
    fd.append(row_d_values2)
fd = np.array(fd)

# prediction
model.eval()
output_list = []
with torch.no_grad():
    for i in range(data.shape[0]):
        data_batch = data[i].unsqueeze(0)
        
        mask_batch = mask_patch[i].unsqueeze(0)
        outputtemp = model(data_batch,mask_batch)
        output_list.append(outputtemp)

output = torch.cat(output_list, dim=0)

row = 825
col = 1310

overlap_grid = np.ceil(nsize * overlap_rate).astype(int)
ndown = nsize - overlap_grid

""" 计算上下左右分别要补多少 """
if (row - overlap_grid) % ndown != 0:
    npatch_row = (np.ceil((row - overlap_grid)/ndown)).astype(int)
    nrow_add = (npatch_row*ndown + overlap_grid - row).astype(int)
    n_up = np.ceil(nrow_add/2).astype(int)
    n_down = (nrow_add - n_up).astype(int)

if (col - overlap_grid) % ndown != 0:
    npatch_col = (np.ceil((col - overlap_grid)/ndown)).astype(int)
    ncol_add = (npatch_col*ndown + overlap_grid - col).astype(int)
    n_left = np.ceil(ncol_add/2).astype(int)
    n_right = (ncol_add - n_left).astype(int)

""" 补齐之后，big_image是fine-resolution, cdimage是coarse-resolution, counter计算每一个点覆盖了几次 """
big_image = np.zeros((row + nrow_add, col + ncol_add))
cdimage = np.zeros((row + nrow_add, col + ncol_add))
counter = np.zeros((row + nrow_add, col + ncol_add))

for idx in range(npatch_row * npatch_col):
    i = idx // npatch_col
    j = idx % npatch_col
    
    start_i = i * ndown
    start_j = j * ndown
    
    patch = output[idx, 0]
    
    big_image[start_i:start_i+nsize, start_j:start_j+nsize] += patch.cpu().numpy()
    cdimage[start_i:start_i+nsize, start_j:start_j+nsize] += dataold[idx, channels_to_normalize[-1], :, :]
    
    counter[start_i:start_i+nsize, start_j:start_j+nsize] += 1

big_image /= counter
cdimage /= counter

if n_up > 0:
    big_image = big_image[n_up:, :]
    cdimage = cdimage[n_up:, :]
if n_down > 0:
    big_image = big_image[:-n_down, :]
    cdimage = cdimage[:-n_down, :]
if n_left > 0:
    big_image = big_image[:, n_left:]
    cdimage = cdimage[:, n_left:]
if n_right > 0:
    big_image = big_image[:, :-n_right]
    cdimage = cdimage[:, :-n_right]
end_time = time.time()

print(f"The code took {end_time - start_time} seconds to run.")

# Analyze accuracy
# 1. Classification performance
# 1.1 RSME
assert big_image.shape == fd.shape == mask.shape == cdimage.shape, "big_image, fd and mask must have the same shape"
dff = fd * mask-big_image
big_image_masked = big_image[mask == 1]
fd_masked = fd[mask == 1]
cd_masked = cdimage[mask == 1]
diff = big_image_masked - fd_masked
diff_sq = diff ** 2
mean_diff_sq = np.sum(diff_sq)/np.sum(mask)
rmse = np.sqrt(mean_diff_sq)
print('Regression results')
print('---------DL---------')
print('RSME: {:.4f}'.format(rmse))

mae = np.sum(np.abs(diff)) / np.sum(mask)
print('MAE: {:.4f}'.format(mae))

mean_observed = np.mean(fd_masked)
sse = np.sum(diff_sq)
sst = np.sum((fd_masked - mean_observed) ** 2)
nse = 1 - (sse / sst)
print('NSE: {:.4f}'.format(nse))

# 1.2 Correlation coefficient
corrcoef = np.corrcoef(big_image_masked.flatten(), fd_masked.flatten())[0, 1]
print("PCC: {:.4f}".format(corrcoef))
print('---------CD---------')
diffcd = cd_masked - fd_masked
diff_sqcd = diffcd ** 2
mean_diff_sqcd = np.sum(diff_sqcd)/np.sum(mask)
rmsecd = np.sqrt(mean_diff_sqcd)
print('RSME: {:.4f}'.format(rmsecd))

maecd = np.sum(np.abs(diffcd)) / np.sum(mask)
print('MAE: {:.4f}'.format(maecd))

mean_observedcd = np.mean(fd_masked)
ssecd = np.sum(diff_sqcd)
sstcd = np.sum((fd_masked - mean_observedcd) ** 2)
nsecd = 1 - (ssecd / sstcd)
print('NSE: {:.4f}'.format(nsecd))

# 1.2 Correlation coefficient
corrcoefcd = np.corrcoef(cd_masked.flatten(), fd_masked.flatten())[0, 1]
print("PCC: {:.4f}".format(corrcoefcd))

# 2. Regression performance
# Accuracy interval
def map_to_interval(x):
    """ 给洪水分类别 """

    if x < 0.05:
        return 1
    elif  0.05<= x < 0.5:
        return 2
    elif  0.5 <= x < 1.5:
        return 3
    else:
        return 4
    
def calculate_precision_recall(y_pred, y_true, class_label):
    tp = np.sum((y_true == class_label) & (y_pred == class_label))
    fp = np.sum((y_true != class_label) & (y_pred == class_label))
    fn = np.sum((y_true == class_label) & (y_pred != class_label))

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    csi = tp / (tp + fn + fp) if tp + fn + fp > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, csi, f1

big_image_intervals = np.array(list(map(map_to_interval, big_image_masked)))
cd_image_intervals = np.array(list(map(map_to_interval, cd_masked)))
fd_intervals = np.array(list(map(map_to_interval, fd_masked)))
same_elements = np.sum(big_image_intervals == fd_intervals)
same_elementscd = np.sum(cd_image_intervals == fd_intervals)
accuracybig = same_elements / len(big_image_intervals)
accuracycd = same_elementscd / len(cd_image_intervals)

# Calculate precision and recall for each class
print('Classification results')
print('---------DL---------')
# Macro-averaging
macro_precision = 0
macro_recall = 0
macro_csi = 0
macro_f1 = 0
num_classes = len(np.unique(big_image_intervals))

micro_tp = 0
micro_fp = 0
micro_fn = 0

for class_label in np.unique(big_image_intervals):
    tp = np.sum((fd_intervals == class_label) & (big_image_intervals == class_label))
    fp = np.sum((fd_intervals != class_label) & (big_image_intervals == class_label))
    fn = np.sum((fd_intervals == class_label) & (big_image_intervals != class_label))
    micro_tp += tp
    micro_fp += fp
    micro_fn += fn

micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp > 0 else 0
micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn > 0 else 0
micro_csi = micro_tp / (micro_tp + micro_fn + micro_fp) if micro_tp + micro_fn + micro_fp > 0 else 0
micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if micro_precision + micro_recall > 0 else 0

print(f"Micro-averaged Precision = {micro_precision:.4f}, Recall = {micro_recall:.4f}, CSI = {micro_csi:.4f}, F1 = {micro_f1:.4f}")
""" for class_label in np.unique(fd_intervals): """
for class_label in np.unique(big_image_intervals):
    precision, recall, csi, f1= calculate_precision_recall(big_image_intervals, fd_intervals, class_label)
    print(f"Class {class_label}: Precision = {precision:.4f}, Recall = {recall:.4f}, CSI = {csi:.4f}, F1 = {f1:.4f}")

print("Accuracy: {:.2%}".format(accuracybig))

print('---------CD---------')
""" for class_label in np.unique(fd_intervals): """
for class_label in np.unique(big_image_intervals):
    precision, recall, csi, f1= calculate_precision_recall(cd_image_intervals, fd_intervals, class_label)
    print(f"Class {class_label}: Precision = {precision:.4f}, Recall = {recall:.4f}, CSI = {csi:.4f}, F1 = {f1:.4f}")

print("Accuracy: {:.2%}".format(accuracycd))