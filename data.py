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


# def prepare_dataloaders(df_pos_train_fold, df_neg_train_fold, all_embeds_pos, all_embeds_neg,
#                         batch_size: int=32, num_workers: int=10,
#                         train_by_all_data: bool=False):
#
    # train_df = dataframe.loc[dataframe['train_or_test'] == 'train']
    # test_df = dataframe.loc[dataframe['train_or_test'] == 'test']
    # val_df = train_df[train_df['val'] == 1]
    # train_df = train_df[train_df['val'] != 1]
    # if train_by_all_data:
    #     train_df = train_df.append([test_df])
    #     train_df = train_df.append([val_df])
    #     test_df = train_df[0:0]
    #     val_df = train_df[0:0]


    # train_image_fname = glob.glob(train_image_path + '/**/*.jpg', recursive=True) + glob.glob(
    #     train_image_path + '/**/*.png', recursive=True) + \
    #                     glob.glob(train_image_path + '/**/*.jpeg', recursive=True)
    #
    # val_image_fname = glob.glob(val_image_path + '/**/*.jpg', recursive=True) + glob.glob(val_image_path + '/**/*.png',
    #                                                                                       recursive=True) + \
    #                   glob.glob(val_image_path + '/**/*.jpeg', recursive=True)
    #
    # train_image_text = [x.split('/')[-2] for x in train_image_fname]
    #
    # val_image_text = [x.split('/')[-2] for x in val_image_fname]

    # train_dataset = DataSet(df_pos_train_fold=df_pos_train_fold, all_embeds_pos=all_embeds_pos,
    #                         df_neg_train_fold=df_neg_train_fold, all_embeds_pos=all_embeds_neg)
    #
    # val_dataset = DataSet(img_preprocess=img_preprocess, tokenizer=tokenizer,
    #                    image_path=val_image_fname, text_list=val_image_text, classifier_uniq_cls=True)
    #
    # train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
    #                                                shuffle=train_dataset.isShuffle,
    #                                                num_workers=num_workers,
    #                                                sampler=train_dataset.sampler)
    #
    # val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
    #                                              shuffle=False,
    #                                              num_workers=num_workers,)
    #
    # return train_dataloader, val_dataloader
