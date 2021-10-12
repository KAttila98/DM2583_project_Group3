#!/usr/bin/env python
# coding: utf-8

import random
import copy
import time
import pandas as pd
import numpy as np
import gc
import re
import torch

#import spacy
from tqdm import tqdm_notebook, tnrange
from tqdm.auto import tqdm

tqdm.pandas(desc='Progress')
from collections import Counter

from nltk import word_tokenize

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from sklearn.metrics import f1_score
import os 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# cross validation and metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.optim.optimizer import Optimizer

from functools import partial
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import re
import string
import pickle

import matplotlib.pyplot as plt

embed_size = 300 # how big is each word vector
n_layers = 1
hidden_size = 128
fcl_size = 64
max_features = 200000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 60 # max number of words in a question to use
batch_size = 512 # how many samples to process at once
n_epochs = 10 # how many times to iterate over all samples

# review_text = "It's the best movie I've ever seen!!! <3"
review_text = str(input("Enter a review text: "))

contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re
contractions, contractions_re = _get_contractions(contraction_dict)
def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

# data preprocessing
def preprocess_text(text):
    
    # converting text to lowercase
    text = text.lower()
    # stop_words = set(stopwords.words('english'))
    text = replace_contractions(text)
    # replacing url-s with the word 'url'
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','', text)
    # replacing usernames-s with the word 'user'
    text = re.sub('@[^\s]+','', text)
    # remove HTML tags
    text = re.sub('<.*?>', '', text)
    # remove punctuation marks
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    # remove multiple spaces
    text = re.sub(' +',' ', text)
    # replacing numbers with the word 'number'
    text = re.sub(r'\w*\d+\w*', '', text)
    
    tokens = word_tokenize(text)
    
    # cleaned_text = [w for w in tokens if not w.lower() in stop_words]
    
    cleaned_text = [w for w in tokens if len(w)>=2]

    # text = deEmojify(text)
    
    return cleaned_text

text_prep = preprocess_text(review_text)
print(f"Preprocessed and tokenized input: {text_prep}")

text_prep = [text_prep]

# loading tokenizer
with open('input/tokenizer_with_stoppwords.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

test_X = tokenizer.texts_to_sequences(list(text_prep))

## Pad the sentences 
test_X = pad_sequences(test_X, maxlen=maxlen)

class LSTM(nn.Module):
    
    def __init__(self):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        drp = 0.2
        n_classes = 3
        self.embedding = nn.Embedding(max_features, embed_size)
        # self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        # self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, self.hidden_size, num_layers = n_layers, batch_first=True)
        # self.linear = nn.Linear(self.hidden_size , fcl_size)
        # self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(self.hidden_size, n_classes)


    def forward(self, x):
        h_embedding = self.embedding(x)
        #_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        h_lstm, _ = self.lstm(h_embedding)
        
        # conc = self.relu(self.linear(h_lstm[:,-1,:]))
        conc = h_lstm[:,-1,:]
        conc = self.dropout(conc)
        out = self.out(conc)
        return out

print("Predicting sentiment...")

model = LSTM()
model.load_state_dict(torch.load("trained_models/lstm_glove200k_10epoch_with_stoppwords"))
model.cuda()

inp = torch.LongTensor(test_X).cuda()
model.eval()
y_pred = model(inp).detach()
pred =F.softmax(y_pred, dim = 1).cpu().numpy()

sentiments = ['negative', 'neutral', 'positive']
prob = np.max(pred)
sentiment = sentiments[np.argmax(pred, axis = 1)[0]]

print(f'The sentence is {int(prob*100)} % likely to be {sentiment}.')



