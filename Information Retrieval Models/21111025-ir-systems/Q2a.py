#!/usr/bin/env python
# coding: utf-8

# In[11]:


import json
import re
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from pathlib import Path
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


def BRSystem(query, w_coll):
    query = query.lower()
    q_words = token_stem(query)
    rel_docs = []
    bitmap = {}
    for path in Path("english-corpora").iterdir():
        fname = path.name[:-4]
        bitmap[fname] = [1]
        for word in q_words:
            if fname in w_coll[word].keys():
                bitmap[fname].append(1)
            else:
                bitmap[fname].append(0)
        b = 1
        for i in bitmap[fname] : b &= i
        if b:
            rel_docs.append(fname)
    return rel_docs


# In[12]:


if sys.argv[1] != '-f':
    query = sys.argv[1]
    w_coll = json.load(open('data.json'))
    print("Most Relevant docs are : " + str(BRSystem(query, w_coll)))


# In[9]:


#print("Most Relevant docs are : " + str(BRSystem(query, w_coll)))


# In[ ]:




