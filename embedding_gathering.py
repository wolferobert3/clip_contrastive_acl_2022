from transformers import CLIPProcessor, CLIPTokenizer, CLIPTextModel, GPT2Model, GPT2Tokenizer
import numpy as np
import pandas as pd
import random
import pickle
from os import path, listdir, remove
import torch
#from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from helper_functions import visualize_pca, cosine_similarity
from csv import reader

clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

#gpt2_model = GPT2Model.from_pretrained('gpt2')
#gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#gpt2_eos = gpt2_tokenizer.eos_token
#gpt2_bos = gpt2_tokenizer.bos_token
#gpt2_model.eval()

#EOS Token only relevant for GPT-2
model = clip_model
tokenizer = clip_tokenizer
HAS_EOS_TOKEN = True
GET_SENTENCE_VECTOR = False
IS_GPT2 = False

#### Gathering sentence embeddings

SOURCE_DIR = f'\\sentence_files\\'
sub_dirs = listdir(SOURCE_DIR)

final_sents = []

for dir_ in sub_dirs:    

    source_dir = path.join(SOURCE_DIR,dir_)
    source_files = listdir(source_dir)

    for file in source_files:

        with open(path.join(source_dir,file),'r',encoding='utf8') as reader:
            sentences = reader.read().split('\n')

        sentences = [i.split('\t') for i in sentences]
        sentences = [i for i in sentences if len(i) > 1]

        sentence_list_0 = [i[0] for i in sentences]
        sentence_list_1 = [i[1] for i in sentences]
        joint_list = sentence_list_0 + sentence_list_1

        if final_sents:
            final_sents.extend(joint_list)
        else:
            final_sents = joint_list

print(len(final_sents))

deduplicated_list = list(set(final_sents))
print(len(deduplicated_list))

semeval_sentences = '\n'.join(deduplicated_list)

with open(f'\\sentence_files\\semeval_sentences.txt','w',encoding='utf8') as writer:
    writer.writelines(semeval_sentences)

tokenized = [tokenizer.encode(sentence) for sentence in semeval_sentences]
token_lengths = [len(i) for i in tokenized]

semeval_sentences = [semeval_sentences[i] for i in range(len(semeval_sentences)) if token_lengths[i] < 77]
tokenized = [tokenized[i] for i in range(len(tokenized)) if token_lengths[i] < 77]

with open(f'\\sentence_files\\semeval_tokenizations.pkl','wb') as pkl_writer:
    pickle.dump(tokenized,pkl_writer)

semeval_writer = '\n'.join(semeval_sentences)

with open(f'\\sentence_files\\semeval_sentences_screened.txt','w',encoding='utf8') as writer:
    writer.writelines(semeval_writer)

BATCH_SIZE = 100
tokenizer.pad_token = tokenizer.eos_token

for i in range(0,len(semeval_sentences),BATCH_SIZE):

    tokenizations = []
    embeddings = []

    start = i
    end = min(i+BATCH_SIZE,len(semeval_sentences))
    targets = semeval_sentences[start:end]

    inputs = tokenizer(targets, max_length=77,truncation=True,padding=True,return_tensors="pt")
    
    with torch.no_grad():
        text_features = model(**inputs,output_hidden_states=True)

    hidden = text_features[2]

    with open(f'\\sentence_files\\semeval_embeddings_{i}.pkl','wb') as pkl_writer:
        pickle.dump(hidden,pkl_writer)