import os
import glob
# import pandas as pd
from PIL import Image
import numpy as np
import tqdm
import sklearn.metrics
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset as Dataset
import torch.nn as nn
import torch.optim as optim

def to_device(tensor_or_list, device):
    if isinstance(tensor_or_list, (list, tuple)):
        tensor_or_list = [tensor.to(device) for tensor in tensor_or_list]
    else:
        tensor_or_list = tensor_or_list.to(device)

    return tensor_or_list

class DataSet(Dataset):
    def __init__(self, df_pos_fold, all_embeds_pos, df_neg_fold, all_embeds_neg, is_train: bool=False, **kwargs):
        super(Dataset, self).__init__()
        # all_embeds_pos['index']
        self.embs = np.concatenate([all_embeds_pos[df_pos_fold['index'], :], all_embeds_neg[df_neg_fold['index'], :]])
        self.labels = np.concatenate([np.ones((1, df_pos_fold['index'].shape[0])),
                                      np.zeros((1, df_neg_fold['index'].shape[0]))], axis=1).astype(np.float32)
        self.labels = self.labels.reshape(-1)
        self.train_dataset = is_train

        if self.train_dataset:
            self.sampler = self._minority_database_oversampling()
        else:
            self.sampler = None

        return

    def __len__(self):
        return self.labels.size

    def __getitem__(self, idx: int):
        return self.embs[idx, :], self.labels[idx]

    def _minority_database_oversampling(self):
        cls_id = np.zeros(self.labels.size).astype(int)
        cls_id[self.labels.reshape(-1)==1] = 1
        class_sample_count = np.array([len(np.where(cls_id == t)[0]) for t in np.unique(cls_id)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in cls_id])
        # replacement=True : If you specify replacement=False and keep the size as the whole dataset length, only your batches at the beginning will
        # be balanced using the weights until all minority classes are “used”. You could try to decrease the length so that most of
        # your batches will be balanced.
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.from_numpy(samples_weight).type(torch.DoubleTensor), len(samples_weight), replacement=True)
        self.isShuffle = False  # shuffle is built in the Weighted Random sample
        return sampler


