#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from num2words import num2words
from pathlib import Path
import json
import re


# In[2]:


ps = PorterStemmer()
from nltk.corpus import stopwords
stopWords = stopwords.words('english')


# In[3]:


w_col = {}


# In[4]:


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


# In[5]:


for path in Path("english-corpora").iterdir():
    fname = path.name[:-4]
    text = open(path, 'r' , encoding="utf8").read()
    words = token_stem(text)
    
    for word in words:
        if word in w_col.keys():
            if fname in w_col[word].keys():
                w_col[word][fname] += 1
            else:
                w_col[word][fname] = 1
        else:
            temp = {fname : 1}
            w_col[word] = temp.copy()


# In[6]:


len(w_col.keys())


# In[7]:


with open('data.json', 'w') as fp:
    json.dump(w_col, fp,  indent=4)


# In[ ]:




