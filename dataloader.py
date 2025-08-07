# Rapid mapping of flood inundation by deep learning-based image super-resolution
# Developer: Wenke Song
# The University of Hong Kong
# Contact email: songwk@connect.hku.hk
# MIT License
# Copyright (c) 2024 songwk0924


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


class NpyDataset(Dataset):
    def __init__(self, feature_path, label_path):

        """ 这是字典序，用正则表达式用自然排序（文件名中的数字排序）"""
        feature_list = sorted(os.listdir(feature_path))
        label_list = sorted(os.listdir(label_path))

        print(f'feature_list: {feature_list}')
        print(f'label_list: {label_list}')

        # Load all feature and label data
        feature = np.concatenate([np.load(os.path.join(feature_path, file)) for file in feature_list], axis=0)
        label = np.concatenate([np.load(os.path.join(label_path, file)) for file in label_list], axis=0)
        # print(f'feature shape: {self.feature.shape}')
        # print(f'label shape: {self.label.shape}')

        # Find indices where feature is not all zeros
        """ 域外值不应该是NAN吗 这里应该额外加一个mask channel """
        valid_indices = np.array([i for i in range(len(feature)) if np.any(feature[i])])

        # Only keep samples where the feature is not all zeros
        self.feature = feature[valid_indices]
        self.label = label[valid_indices]

        # Normalize the feature data
        """ 这里发生了验证集的泄露吧，计算min max的时候不应该包含验证集 """
        channels_to_normalize = [0, 1, 2, 3, 4]
        for c in channels_to_normalize:
            min_val = self.feature[:, c, ...].min(axis=(0, 1, 2), keepdims=True)
            max_val = self.feature[:, c, ...].max(axis=(0, 1, 2), keepdims=True)
            print(min_val)
            print(max_val)

            if c == 2:
                self.feature[:, c, ...] = (self.feature[:, c, ...]) / 6
            else:
                self.feature[:, c, ...] = (self.feature[:, c, ...] - min_val) / (max_val - min_val)

        print('max: ', self.feature[:, 0, :, :].max(), self.feature[:, 1, :, :].max(), self.feature[:, 2, :, :].max(),
              self.feature[:, 3, :, :].max(), self.feature[:, 4, :, :].max())
        print('min: ', self.feature[:, 0, :, :].min(), self.feature[:, 1, :, :].min(), self.feature[:, 2, :, :].min(),
              self.feature[:, 3, :, :].min(), self.feature[:, 4, :, :].min())

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        """ 写法有点冗余，一共应该是有8种数据增强，无翻转+0 无翻转+90 无翻转+180 无翻转+270 H翻转+0 V翻转+0 H翻转+90 H翻转+270
       目前的写法并不是均匀采样这八种，因此需要修改一下 """

        feature_data = self.feature[idx]
        label_data = self.label[idx]

        # Apply data augmentation
        if np.random.rand() < 0.5:
            # Random horizontal flip
            feature_data = feature_data[:, :, ::-1].copy()
            label_data = label_data[:, :, ::-1].copy()

        if np.random.rand() < 0.5:
            # Random vertical flip
            feature_data = np.flip(feature_data, axis=1).copy()
            label_data = np.flip(label_data, axis=1).copy()

        # Random rotation by 0, 90, 180, or 270 degrees
        num_rotations = np.random.randint(4)
        feature_data = np.rot90(feature_data, num_rotations, axes=(1, 2)).copy()
        label_data = np.rot90(label_data, num_rotations, axes=(1, 2)).copy()

        return torch.from_numpy(feature_data).float(), torch.from_numpy(label_data).float()
