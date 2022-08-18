#!/usr/bin/env python
# coding: utf-8

# In[19]:


import json
import re
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from pathlib import Path
import numpy as np
from num2words import num2words
import sys


# In[2]:


def token_stem(string):
    ps = PorterStemmer()
    from nltk.corpus import stopwords
    stopWords = stopwords.words('english')
    string = string.lower()
    string = re.sub(r'[^\w\s]', ' ', string)                     #To remove punctuations
    string = string.encode("ascii", "ignore").decode()          #To remove non-ASCII characters
    tokens = nltk.word_tokenize(string)
    #Numeric to Alpha-numeric
    for i in range(len(tokens)):
        if tokens[i].isnumeric():
            num = re.sub(r'[^\w\s]', ' ', num2words(tokens[i])).split()
            tokens.pop(i)
            tokens += num
    tokens = [ps.stem(word) for word in tokens if word not in stopWords]
    return tokens


# In[31]:


def BM25(query, w_coll, l = 10, b = 0.75, k = 2):
    query = query.lower()
    q_tokens = token_stem(query)
    lengths = {}
    N = 0
    avg_len = 0
    for path in Path("english-corpora").iterdir():
        fname = path.name[:-4]
        lengths[fname] = len(open(path, 'r' , encoding="utf8").read().split())             #cal no of words of each file
        N += 1                          #To calculate total no of files
        avg_len += lengths[fname]
    avg_len /= N
    #Calculate idf of each token
    idf = {}
    for word in np.unique(q_tokens):
        if word in w_coll.keys():
            df = len(w_coll[word].keys())
        else:
            df = 0
        idf[word] = np.log((N - df + 0.5) / (df + 0.5))
    score = {}
    for path in Path("english-corpora").iterdir():
        fname = path.name[:-4]
        s = 0
        for word in np.unique(q_tokens):
            tf = 0
            if word in w_coll.keys() and fname in w_coll[word].keys():
                tf = w_coll[word][fname]
            s += idf[word] * (tf * (k + 1)) / (k*(1 - b + b*lengths[fname]/avg_len) + tf)
        score[fname] = s
    return sorted(score, key = score.get, reverse=True)[:l]


# In[25]:


if sys.argv[1] != '-f':
    query = sys.argv[1]
    w_coll = json.load(open('data.json'))
    print("Most Relevant docs are : " + str(BM25(query, w_coll)))


# In[26]:


#print("Most Relevant docs are : " + str(BM25(query, w_coll)))


# In[ ]:




