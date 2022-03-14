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
from csv import reader

clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

gpt2_model = GPT2Model.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_eos = gpt2_tokenizer.eos_token
gpt2_bos = gpt2_tokenizer.bos_token
gpt2_model.eval()

#EOS Token only relevant for GPT-2
model = clip_model
tokenizer = clip_tokenizer
HAS_EOS_TOKEN = True
GET_SENTENCE_VECTOR = False
IS_GPT2 = False

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

#### Intrinsic Evaluations - Word Level

EVALUATION = f'rg65.csv'

intrinsic_evaluation = pd.read_csv(f'\\Documents\\{EVALUATION}',index_col=None,sep=';',header=None)
words_left= intrinsic_evaluation[0].tolist()
words_right = intrinsic_evaluation[1].tolist()
relatedness = intrinsic_evaluation[2].tolist()

#if HAS_EOS_TOKEN and IS_GPT2:
    #words_left = [f'{gpt2_bos}{word}{gpt2_eos}' for word in words_left]
    #words_right = [f'{gpt2_bos}{word}{gpt2_eos}' for word in words_right]

words_left_embeddings = get_embedding_list(words_left,get_sentence_vec=GET_SENTENCE_VECTOR,has_eos_token=HAS_EOS_TOKEN)
words_right_embeddings = get_embedding_list(words_right,get_sentence_vec=GET_SENTENCE_VECTOR,has_eos_token=HAS_EOS_TOKEN)

layerwise_scores = []

for layer in range(13):
    cosine_similarities = [cosine_similarity(words_left_embeddings[i][layer],words_right_embeddings[i][layer]) for i in range(len(words_right_embeddings))]
    evaluation_score = spearmanr(cosine_similarities,relatedness)
    layerwise_scores.append(evaluation_score[0])

print(layerwise_scores)
val_str = ' '.join([f'({i},{layerwise_scores[i]})' for i in range(len(layerwise_scores))])
print(val_str)