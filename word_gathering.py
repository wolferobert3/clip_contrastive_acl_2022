from transformers import CLIPProcessor, CLIPTokenizer, CLIPTextModel
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random
import pickle
from os import path, listdir, remove
from scipy.spatial.distance import cdist, pdist, correlation, cosine
from scipy.stats import pearsonr, spearmanr
from collections import Counter
import torch
#from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression
from helper_functions import visualize_pca, cosine_similarity
from csv import reader

#### Gather word embeddings
"""
CLIP_SOURCE = f'\\sentence_files\\semeval_clip_embeddings'
clip_files = [i for i in listdir(CLIP_SOURCE) if '38300' not in i]

with open(f'\\sentence_files\\semeval_tokenizations_clip.pkl','rb') as pkl_reader:
    clip_tokenizations = pickle.load(pkl_reader)

tokenization_dict = {idx:len(vecs) for idx, vecs in enumerate(clip_tokenizations)}

for f in clip_files:

    with open(path.join(CLIP_SOURCE,f),'rb') as pkl_reader:
        emb_obj = pickle.load(pkl_reader)
    
    base_num = f[19:-4]
    indices = [random.randint(1,tokenization_dict[int(base_num)+j]) for j in range(100)]

    layer_embs = []

    for i in range(len(emb_obj)):

        layer_arr = np.array([emb_obj[i][j].numpy()[indices[j]] for j in range(len(indices))])
        layer_embs.append(layer_arr)

        with open(f'\\sentence_files\\clip_layers\\{f}_layer_{i}.pkl','wb') as pkl_writer:
            pickle.dump(layer_arr,pkl_writer)


layer_embs = [i for i in listdir(f'\\sentence_files\\gpt2_layers') if i[-3:] == 'pkl']
under_set = list(set([i.split('.')[0][19:] for i in layer_embs]))
random.shuffle(under_set)
under_set = under_set[:100]
final_embs = [f'semeval_embeddings_{i}.pkl_layer_{j}.pkl' for i in under_set for j in range(13)]

for j in range(0,10):
    arr_list = []
    current_ = [i for i in final_embs if i[-5] == str(j) and i[-6] != '1']
    print(current_)
    for f in current_:
        with open(path.join(f'\\sentence_files\\gpt2_layers\\',f),'rb') as pkl_reader:
            arr = pickle.load(pkl_reader)
        if arr_list:
            arr_list.append(arr)
        else:
            arr_list = [arr]
    final_arr = np.vstack(arr_list)
    with open(f'\\sentence_files\\gpt2_layers\\full_layers\\layer_{j}.pkl','wb') as pkl_writer:
        pickle.dump(final_arr,pkl_writer)

for j in range(10,13):
    arr_list = []
    current_ = [i for i in final_embs if i[-5] == str(j)[1] and i[-6] == '1']
    print(current_)
    for f in current_:
        with open(path.join(f'\\sentence_files\\gpt2_layers\\',f),'rb') as pkl_reader:
            arr = pickle.load(pkl_reader)
        if arr_list:
            arr_list.append(arr)
        else:
            arr_list = [arr]
    final_arr = np.vstack(arr_list)
    with open(f'\\sentence_files\\gpt2_layers\\full_layers\\layer_{j}.pkl','wb') as pkl_writer:
        pickle.dump(final_arr,pkl_writer)
"""

#Geometric Analysis of Embeddings

#gpt_source = f'\\sentence_files\\gpt2_layers\\full_layers'
clip_source = f'\\sentence_files\\clip_layers\\full_layers'

#gpt_files = listdir(gpt_source)
clip_files = listdir(clip_source)

self_sims = []
norms = []
tops = []
pcts = []
iso_sims = []

for idx, f in enumerate(clip_files):
    with open(path.join(clip_source,f),'rb') as pkl_reader:
        arr = pickle.load(pkl_reader)

    sim = np.mean(1-pdist(arr,metric='cosine'))
    mag = np.mean(np.linalg.norm(arr,axis=1,keepdims=True))
    sorted_row_idx = np.argsort(arr, axis=1)[:,arr.shape[1]-5::]

    col_idx = np.arange(arr.shape[0])[:,None]

    large = arr[col_idx,sorted_row_idx]
    print(large.shape)
    top_ = np.mean(np.linalg.norm(large,axis=1,keepdims=True))
    pct = top_/mag
    print(pct)

    mean_vector = np.mean(arr,axis=0)
    iso_vecs = arr - mean_vector
    iso_sim = np.mean(1-pdist(iso_vecs,metric='cosine'))

    self_sims.append(sim)
    norms.append(mag)
    tops.append(top_)
    pcts.append(pct)
    iso_sims.append(iso_sim)

print(self_sims)
print(norms)
print(tops)
print(pcts)
print(iso_sims)