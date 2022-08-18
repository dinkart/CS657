#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[3]:


def tfidf(query, w_coll, k = 10):
    query = query.lower()
    q_tokens = token_stem(query)
    lengths = {}
    N = 0
    for path in Path("english-corpora").iterdir():
        fname = path.name[:-4]
        lengths[fname] = len(open(path, 'r' , encoding="utf8").read().split())             #cal no of words of each file
        N += 1                          #To calculate total no of files
    #Calculate idf of each file
    idf = {}
    for word in np.unique(q_tokens):
        df = len(w_coll[word].keys())
        idf[word] = np.log((N + 1) / (df + 1))
    #Calculate tf-idf score vector of each file
    tf_idf = {}
    for path in Path("english-corpora").iterdir():
        fname = path.name[:-4]
        for word in np.unique(q_tokens):
            tf = 0
            if fname in w_coll[word].keys():
                tf = w_coll[word][fname] / lengths[fname]
            if fname not in tf_idf.keys():
                tf_idf[fname] = []
            tf_idf[fname].append(tf * idf[word])
    #Calculate tf-idf score vector of query
    len_q = len(query.split())
    tf_idf_q = []
    for word in np.unique(q_tokens):
        tf = q_tokens.count(word) / len_q
        tf_idf_q.append(tf * idf[word])
    #Calculation of cosine-similarity
    cos_sim = {}
    for path in Path("english-corpora").iterdir():
        fname = path.name[:-4]
        den = (np.linalg.norm(tf_idf[fname])*np.linalg.norm(tf_idf_q))      #denominator
        if not den:
            cos_sim[fname] = 0
        else:
            cos_sim[fname] = np.dot(tf_idf[fname], tf_idf_q)/den
    #return list of k most relevant files
    return sorted(cos_sim, key = cos_sim.get, reverse=True)[:k]


# In[11]:


if sys.argv[1] != '-f':
    query = sys.argv[1]
    w_coll = json.load(open('data.json'))
    print("Most Relevant docs are : " + str(tfidf(query, w_coll)))


# In[8]:


#print("Most Relevant docs are : " + str(tfidf(query, w_coll)))


# In[10]:


sys.argv[1]


# In[ ]:




