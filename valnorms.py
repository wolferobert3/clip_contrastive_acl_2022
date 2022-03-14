from transformers import CLIPProcessor, CLIPTokenizer, CLIPTextModel, GPT2Tokenizer, GPT2Model
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

#clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
#clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
#clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#clip_model.eval()

gpt2_model = GPT2Model.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_eos = gpt2_tokenizer.eos_token
gpt2_bos = gpt2_tokenizer.bos_token
gpt2_model.eval()

#EOS Token only relevant for GPT-2
model = gpt2_model
tokenizer = gpt2_tokenizer
HAS_EOS_TOKEN = True
GET_SENTENCE_VECTOR = True
IS_GPT2 = True

def get_embedding_list(string_list,get_sentence_vec=False,has_eos_token=True):

    extraction_id = -2
    if get_sentence_vec or not has_eos_token:
        extraction_id = -1

    returned_embeddings = []

    for string in string_list:
        inputs = tokenizer([string],return_tensors="pt",add_special_tokens=True)
        
        with torch.no_grad():
            text_features = model(**inputs,output_hidden_states=True)

        hidden_states = text_features[2]

        embeddings = []

        for layer_vectors in hidden_states:
        
            if get_sentence_vec:
                representation = layer_vectors[-1].numpy()[extraction_id]
            else:
                representation = layer_vectors[-1].numpy()[extraction_id]
            
            if embeddings:
                embeddings.append(representation)
            else:
                embeddings = [representation]
        
        if returned_embeddings:
            returned_embeddings.append(embeddings)
        else:
            returned_embeddings = [embeddings]

    return returned_embeddings

pleasant = sorted(list(set('caress,freedom,health,love,peace,cheer,friend,heaven,loyal,pleasure,diamond,gentle,honest,lucky,rainbow,diploma,gift,honor,miracle,sunrise,family,happy,laughter,paradise,vacation'.split(','))))
unpleasant = sorted(list(set('abuse,crash,filth,murder,sickness,accident,death,grief,poison,stink,assault,disaster,hatred,pollute,tragedy,divorce,jail,poverty,ugly,cancer,kill,rotten,vomit,agony,prison'.split(','))))

bellezza = pd.read_csv(f'\\Bellezza_Lexicon.csv',index_col='word')
bellezza_words = bellezza.index.tolist()
bellezza_valence = bellezza['combined_pleasantness'].tolist()

if HAS_EOS_TOKEN and IS_GPT2:
    pleasant = [f'{gpt2_bos}{word}{gpt2_eos}' for word in pleasant]
    unpleasant = [f'{gpt2_bos}{word}{gpt2_eos}' for word in unpleasant]
    bellezza_words = [f'{gpt2_bos}{word}{gpt2_eos}' for word in bellezza_words]

pleasant_embs = get_embedding_list(pleasant,get_sentence_vec=GET_SENTENCE_VECTOR,has_eos_token=HAS_EOS_TOKEN)
pleasant_dict = {pleasant[i]:pleasant_embs[i] for i in range(len(pleasant))}

unpleasant_embs = get_embedding_list(unpleasant,get_sentence_vec=GET_SENTENCE_VECTOR,has_eos_token=HAS_EOS_TOKEN)
unpleasant_dict = {unpleasant[i]:unpleasant_embs[i] for i in range(len(unpleasant))}

bellezza_embs = get_embedding_list(bellezza_words,get_sentence_vec=GET_SENTENCE_VECTOR,has_eos_token=HAS_EOS_TOKEN)
bellezza_dict = {bellezza_words[i]:bellezza_embs[i] for i in range(len(bellezza_words))}

valnorms = []

for i in range(0,13):
    pleasant_ = [word_emb[i] for word_emb in pleasant_embs]
    unpleasant_ = [word_emb[i] for word_emb in unpleasant_embs]
    bellezza_ = [word_emb[i] for word_emb in bellezza_embs]

    pleasant_arr = 1 - cdist(bellezza_,pleasant_,metric='cosine')
    unpleasant_arr = 1 - cdist(bellezza_,unpleasant_,metric='cosine')
    full_arr = np.concatenate((pleasant_arr,unpleasant_arr),axis=1)

    pleasant_mean = np.mean(pleasant_arr,axis=1)
    unpleasant_mean = np.mean(unpleasant_arr,axis=1)
    joint_std = np.std(full_arr,axis=1)

    weat_diff = pleasant_mean - unpleasant_mean
    weats = weat_diff / joint_std
    valnorms.append(pearsonr(weats,bellezza_valence))

print(valnorms)
plt.plot(list(range(13)),valnorms)
plt.show()