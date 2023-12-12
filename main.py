import sys

import os
from scipy.optimize import linear_sum_assignment
import pickle
# import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('qtagg')
import matplotlib.pyplot as plt
import re
import copy

import numpy as np
import torch
import tqdm
# np.seterr(all='raise')
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import emoji
# from abbrev_dict import *
# from eval_metrices import roc_plot, p_r_plot
from data import *
from models import *
from train_eval import *
def flatten(lst):
    return [x for l in lst for x in l]

def cosine_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x)*np.linalg.norm(y))

class SimilarityManager:
    def __init__(self, device):
        # self.similarity_model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.similarity_model.to(device)

class VGEvaluation:
    def __init__(self, device):
        self.smanager = SimilarityManager(device)

    def encode_tokens (self, t1: list, batch_size=16, show_progress_bar=True):
        embs = self.smanager.similarity_model.encode([t.lower() for t in t1],
                                                     batch_size=batch_size,
                                                     show_progress_bar=show_progress_bar)
        return embs

    def sm_similarity(self, t1: str, t2: str):
        embs = self.smanager.similarity_model.encode([t1.lower(), t2.lower()])
        sim = cosine_sim(*embs)
        return sim

    def compute_scores(self, src, dst,
                       **kwargs):  # sm = sm_similarity(tuple([obj_gt['label']]), tuple([obj_det['label']])) # get into the format of ('token',)
        scores_matrix = [[self.sm_similarity(x, y) for y in dst] for x in src]
        return np.array(scores_matrix)

    def compute_scores(self, src_embed, dst_embed,
                       **kwargs):  # sm = sm_similarity(tuple([obj_gt['label']]), tuple([obj_det['label']])) # get into the format of ('token',)
        scores_matrix = [[cosine_sim(x, y) for y in dst_embed] for x in src_embed]
        return np.array(scores_matrix)

def embeddings_extract(sentence: list, key_tag:str , result_dir: str, evaluator):
    batch_size = 32

    if len(sentence) % batch_size != 0:  # all images size are Int multiple of batch size
        pad = batch_size - len(sentence) % batch_size
    else:
        pad = 0

    all_embeds = list()
    bns = len(sentence)//batch_size

    for idx in np.arange(bns):
        batch_sent = sentence[idx * batch_size: (idx + 1) * batch_size]
        batch_sent = [re.sub('\r\n', '', dialog) for dialog in batch_sent] # TODO HK@@
        embs = evaluator.encode_tokens(batch_sent)
        all_embeds.append(embs)

        if idx % 10 == 0:
            with open(os.path.join(result_dir, str(key_tag) + '.pkl'), 'wb') as f:
                pickle.dump(all_embeds, f)
    if pad != 0:
        batch_sent = sentence[batch_size * (len(sentence)//batch_size): len(sentence)]
        embs = evaluator.encode_tokens(batch_sent)
        all_embeds.append(embs)

    all_embeds = np.concatenate(all_embeds)

    with open(os.path.join(result_dir, str(key_tag) + '.pkl'), 'wb') as f:
        pickle.dump(all_embeds, f)

    return all_embeds

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    num_workers = 4

    evaluator = VGEvaluation(device)
    local_dir = r'C:\Users\h00633314\HanochWorkSpace\Projects\Isis_tweet_detection'
    bin_dir = os.path.join(local_dir, 'bin')
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    # Model
    embeddings = evaluator.smanager.similarity_model.encode("This is an example sentence")
    model = Mlp(in_features=embeddings.shape[0], out_features=2, drop=0)
    model.to(device)


    pre_compute_embeddings = False

    data_dir = os.path.join(local_dir, 'data')

    df_neg, df_pos = load_preprocess_csv_data(data_dir)

    # evaluator.encode_tokens
    if pre_compute_embeddings:
        print("precompute embeddings saved to pickles")

        all_embeds = evaluator.encode_tokens(df_pos['tweet_post_proc'].to_list())

        key_tag = 'isis_pos_embed'
        with open(os.path.join(bin_dir, str(key_tag) + '.pkl'), 'wb') as f:
            pickle.dump(all_embeds, f)

        all_embeds = evaluator.encode_tokens(df_neg['tweet_post_proc'].to_list())

        key_tag = 'rand_neg_embed'
        with open(os.path.join(bin_dir, str(key_tag) + '.pkl'), 'wb') as f:
            pickle.dump(all_embeds, f)
    else:
        key_tag = 'isis_pos_embed'
        with open(os.path.join(bin_dir, str(key_tag) + '.pkl'), 'rb') as f:
            all_embeds_pos = pickle.load(f)

        key_tag = 'rand_neg_embed'
        with open(os.path.join(bin_dir, str(key_tag) + '.pkl'), 'rb') as f:
            all_embeds_neg = pickle.load(f)

    # Optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5,
                            betas=(0.9, 0.98), eps=1e-6,
                            weight_decay=0.0001) # the lr is smaller, more safe for fine tuning to new dataset

    # [len(df_pos[df_pos.username==x]) for x in np.unique(df_pos.username)]  between 1-1580 tweets per username
    test_train_ratio = 10
    # Stratification by username over the positives , negatives doesn't have that info
    len(df_pos) // test_train_ratio
    pos_inx_users_for_test_set_indices = np.where(np.cumsum([len(df_pos[df_pos.username==x]) for x in np.unique(df_pos.username)]) <len(df_pos) // test_train_ratio)
    pos_inx_users_for_trainval_set_indices = np.where(np.cumsum([len(df_pos[df_pos.username==x]) for x in np.unique(df_pos.username)]) >len(df_pos) // test_train_ratio)

    pos_users_for_test_set_indices = np.unique(df_pos.username)[pos_inx_users_for_test_set_indices]
    pos_users_for_trainval_set_indices = np.unique(df_pos.username)[pos_inx_users_for_trainval_set_indices]

    assert (pos_inx_users_for_test_set_indices[0].shape[0] + pos_inx_users_for_trainval_set_indices[0].shape[
        0] == np.unique(df_pos.username).size)

    df_pos_test = pd.DataFrame()
    for x in pos_users_for_test_set_indices:
        df_pos_test = pd.concat([df_pos_test, df_pos[df_pos['username'] == x]])

    df_pos_trainval = pd.DataFrame()
    for x in pos_users_for_trainval_set_indices:
        df_pos_trainval = pd.concat([df_pos_trainval, df_pos[df_pos['username'] == x]])

    assert (len(df_pos_trainval) + len(df_pos_test) == len(df_pos))

    skf = StratifiedKFold(n_splits=5)
    target = df_pos_trainval.loc[:, 'username']
    skf_gen = skf.split(df_pos_trainval, target)

    df_neg_test = df_neg[-len(df_neg)//test_train_ratio +1 :]
    df_neg_train = df_neg[:-len(df_neg)//10 +1]

    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    for train_index, val_index in kf.split(df_neg_train):
        df_neg_train_fold, df_neg_val_fold = df_neg_train.iloc[train_index, :], df_neg_train.iloc[val_index, :]
        pos_indexes = next(skf_gen)
        (train_pos_index, val_pos_index) = pos_indexes
        df_pos_train_fold, df_pos_val_fold = df_pos_trainval.iloc[train_pos_index, :], df_pos_trainval.iloc[val_pos_index, :]

        train_dataset = DataSet(df_pos_fold=df_pos_train_fold, all_embeds_pos=all_embeds_pos,
                                df_neg_fold=df_neg_train_fold, all_embeds_neg=all_embeds_neg, is_train=True)

        val_dataset = DataSet(df_pos_fold=df_pos_val_fold, all_embeds_pos=all_embeds_pos,
                                df_neg_fold=df_neg_train_fold, all_embeds_neg=all_embeds_neg)


        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=num_workers,
                                                       sampler=train_dataset.sampler)

        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=num_workers)

        all_targets, all_predictions = train_model(model, train_dataloader,
                                                   loss_img=loss, optimizer=optimizer, device=device)

        # prepare_dataloaders(df_pos_train_fold, df_neg_train_fold, df_pos_val_fold,
        #                     df_neg_val_fold,all_embeds_pos, all_embeds_neg)
    pass


def load_preprocess_csv_data(data_dir: str):
    df_pos = pd.read_excel(os.path.join(data_dir, 'tweets_isis_all.xlsx'))
    df_pos['tweet_post_proc'] = df_pos.apply(
        lambda x: re.sub('\n', ' ', x['tweets']), axis=1)
    df_pos['tweet_post_proc'] = df_pos['tweet_post_proc'].apply(
        lambda x: re.sub('\n', ' ', deEmojify(x)))
    df_pos['tweet_post_proc'] = df_pos['tweet_post_proc'].apply(
        lambda x: x.strip())
    df_neg = pd.read_excel(os.path.join(data_dir, 'tweets_random_all.xlsx'))
    df_neg['tweet_post_proc'] = df_neg.apply(
        lambda x: (str(x['content']) + str(x['Unnamed: 2'])).replace("\\", "").replace("\'", " ").strip(), axis=1)
    df_neg['tweet_post_proc'] = df_neg['tweet_post_proc'].apply(
        lambda x: re.sub('\n', ' ', deEmojify(x)))

    df_neg['index'] = range(0, len(df_neg))
    df_pos['index'] = range(0, len(df_pos))

    return df_neg, df_pos


if __name__ == '__main__':
    main()


"""
df_pos['tweet_post_proc'][149] = 'hīs'
df_pos['tweet_post_proc'][428-2] = 'A B'  
if len(df_pos['tweet_post_proc'][149]) <=3 remove tweet!!!
[print(x) for ix, x in enumerate(df_neg['tweet_post_proc']) if ix <30]

# TODO:  tSNE 
"""