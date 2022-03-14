from transformers import CLIPProcessor, CLIPTokenizer, CLIPTextModel, GPT2Model, GPT2Tokenizer
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

#### Intrinsic Evals - Sentence Level

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
model = gpt2_model
tokenizer = gpt2_tokenizer
HAS_EOS_TOKEN = False
GET_SENTENCE_VECTOR = False
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

#### Intrinsic Evaluations - Sentence Level

sts_benchmark = []

with open(f'\\stsbenchmark\\sts-test.csv','r') as read_obj:
    for i in reader(read_obj):
        sts_benchmark.append(i)

#Correct issues in the csv file
sts_benchmark = [''.join(i) for i in sts_benchmark if type(i) == list]

#Prep lists of sentences
sts_benchmark = [i.split('\t') for i in sts_benchmark]
sts_benchmark = [i[4:7] for i in sts_benchmark]
sts_benchmark_df = pd.DataFrame(data=sts_benchmark,columns=['score','sent1','sent2'])

left_sentences = sts_benchmark_df['sent1'].tolist()
right_sentences = sts_benchmark_df['sent2'].tolist()
relatedness = sts_benchmark_df['score'].tolist()

if HAS_EOS_TOKEN and IS_GPT2:
    left_sentences = [f'{gpt2_bos}{sentence}{gpt2_eos}' for sentence in left_sentences]
    right_sentences = [f'{gpt2_bos}{sentence}{gpt2_eos}' for sentence in right_sentences]

sentences_left_embeddings = get_embedding_list(left_sentences,get_sentence_vec=GET_SENTENCE_VECTOR,has_eos_token=HAS_EOS_TOKEN)
sentences_right_embeddings = get_embedding_list(right_sentences,get_sentence_vec=GET_SENTENCE_VECTOR,has_eos_token=HAS_EOS_TOKEN)

layerwise_scores = []

for layer in range(13):
    cosine_similarities = [cosine_similarity(sentences_left_embeddings[i][layer],sentences_right_embeddings[i][layer]) for i in range(len(sentences_right_embeddings))]
    evaluation_score = spearmanr(cosine_similarities,relatedness)
    layerwise_scores.append(evaluation_score[0])

print(layerwise_scores)

#Sentence-Level Self-Similarity
all_unique_sentences = list(set(left_sentences + right_sentences))
all_unique_embeddings = get_embedding_list(left_sentences,get_sentence_vec=GET_SENTENCE_VECTOR,has_eos_token=HAS_EOS_TOKEN)
print(len(all_unique_sentences))

self_similarities = []

for layer in range(13):
    arr = np.array([i[layer] for i in all_unique_embeddings])
    self_sim = np.mean(1-pdist(arr,metric='cosine'))
    self_similarities.append(self_sim)

print(self_similarities)