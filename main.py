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

import pandas as pd
import emoji
# from abbrev_dict import *
# from eval_metrices import roc_plot, p_r_plot
def flatten(lst):
    return [x for l in lst for x in l]

def cosine_sim(x,y):
    return np.dot(x,y) / (np.linalg.norm(x)*np.linalg.norm(y))

class SimilarityManager:
    def __init__(self):
        # self.similarity_model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

        if torch.cuda.device_count() > 0:
            self.similarity_model.cuda()

class VGEvaluation:
    def __init__(self):
        self.smanager = SimilarityManager()

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

    all_embeds_dialog = list()
    bns = len(sentence)//batch_size

    for idx in np.arange(bns):
        batch_sent = sentence[idx * batch_size: (idx + 1) * batch_size]
        batch_sent = [re.sub('\r\n', '', dialog) for dialog in batch_sent] # TODO HK@@
        embs = evaluator.encode_tokens(batch_sent)
        all_embeds_dialog.append(embs)

        if idx % 10 == 0:
            with open(os.path.join(result_dir, str(key_tag) + '.pkl'), 'wb') as f:
                pickle.dump(all_embeds_dialog, f)
    if pad != 0:
        batch_sent = sentence[batch_size * (len(sentence)//batch_size): len(sentence)]
        embs = evaluator.encode_tokens(batch_sent)
        all_embeds_dialog.append(embs)

    all_embeds_dialog = np.concatenate(all_embeds_dialog)

    with open(os.path.join(result_dir, str(key_tag) + '.pkl'), 'wb') as f:
        pickle.dump(all_embeds_dialog, f)

    return all_embeds_dialog

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


def main():
    evaluator = VGEvaluation()

    local_dir = r'C:\Users\h00633314\HanochWorkSpace\Projects\Isis_tweet_detection'
    bin_dir = os.path.join(local_dir, 'bin')
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    data_dir = os.path.join(local_dir, 'data')
    df_pos = pd.read_excel(os.path.join(data_dir, 'tweets_isis_all.xlsx'))


    df_pos['tweet_post_proc'] = df_pos.apply(
        lambda x: re.sub('\n', ' ', x['tweets']), axis=1)

    df_pos['tweet_post_proc'] = df_pos['tweet_post_proc'].apply(
        lambda x: re.sub('\n', ' ', deEmojify(x)))


    df_neg = pd.read_excel(os.path.join(data_dir, 'tweets_random_all.xlsx'))
    df_neg['tweet_merged'] = df_neg.apply(
        lambda x: (str(x['content']) + str(x['Unnamed: 2'])).replace("\\", "").replace("\'", " "), axis=1)

    pre_compute_embeddings = True
    # evaluator.encode_tokens
    if pre_compute_embeddings:
        print("precompute embeddings saved to pickles")

        all_embeds_dialog = evaluator.encode_tokens(df_pos['tweets'].to_list())

        key_tag = 'isis_pos_embed'
        with open(os.path.join(bin_dir, str(key_tag) + '.pkl'), 'wb') as f:
            pickle.dump(all_embeds_dialog, f)


    # all_embds_chunks = embeddings_extract(chunk_list,
    #                                       key_tag=key_tag_chunk,
    #                                       result_dir=result_dir,
    #                                       evaluator=evaluator)

    pass

if __name__ == '__main__':
    main()
