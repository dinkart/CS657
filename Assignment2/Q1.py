#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models import Word2Vec
import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv(r"data/hindi.txt", header = None)


# In[3]:


data


# In[4]:


def norm(v1):
    num = 0
    for i in v1:
        num += i**2
    return num**0.5
def cosine_sim(v1, v2):
    num = 0
    for i,j in zip(v1, v2):
        num += i*j
    return num/(norm(v1)*norm(v2))
    


# In[5]:


thresholds = [4, 5, 6, 7, 8]
dim = [50, 100] #, 200, 300]
models = ["cbow", "fasttext", "sg"]


# In[6]:


for m, d in [(m,d) for m in models for d in dim]:
    model = Word2Vec.load(f"hi\\{d}\\{m}\\hi-d{d}-m2-{m}.model")
    for t in thresholds:
        output = {"word1": [], "word2": [], "Similarity Score": [], "Ground Truth similarity score": [], "Label": []}
        for idx, row in data.iterrows():
            output["word1"].append(row[0])
            output["word2"].append(row[1])
            output["Ground Truth similarity score"].append(row[2])
            
            v1 = model.wv[row[0]]
            v2 = model.wv[row[1]]
            s = cosine_sim(v1,v2)*10
            output["Similarity Score"].append("%.2f" % s)
            if(s >= t):
                output["Label"].append(1)
            else:
                output["Label"].append(0)
        output = pd.DataFrame.from_dict(output)

        #Calculating Accuracy
        ground_trth = []
        for i in data[2]:
            if i >= t:
                ground_trth.append(1)
            else:
                ground_trth.append(0)
        c = 0
        for i, j in zip(output["Label"], ground_trth):
            if i == j:
                c += 1
        a = c/len(output)

        output.to_csv(f'Q1_{d}_{m}_similarity_{t}.csv')
        output = {}
        with open(f'Q1_{d}_{m}_similarity_{t}.csv', 'a') as f:
            try:
                f.write(f"\nAccuracy = {'%.2f' % a}")
            except:
                pass
            f.close()


# For GloVe Data

# In[7]:


for d in dim:
    file = open(f"hi\\{d}\\glove\\hi-d{d}-glove.txt", encoding = "UTF-8", errors = "ignore")
    model = {}
    while True:
        line = file.readline().split()
        if not line:
            break
        model[line[0]] = np.array([float(line[i]) for i in range(1, len(line))])
    for t in thresholds:
        output = {"word1": [], "word2": [], "Similarity Score": [], "Ground Truth similarity score": [], "Label": []}
        for idx, row in data.iterrows():
            output["word1"].append(row[0])
            output["word2"].append(row[1])
            output["Ground Truth similarity score"].append(row[2])

            v1 = model[row[0]]
            v2 = model[row[1]]
            s = cosine_sim(v1,v2)*10
            output["Similarity Score"].append("%.2f" % s)
            if(s >= t):
                output["Label"].append(1)
            else:
                output["Label"].append(0)
        output = pd.DataFrame.from_dict(output)

        #Calculating Accuracy
        ground_trth = []
        for i in data[2]:
            if i >= t:
                ground_trth.append(1)
            else:
                ground_trth.append(0)
        c = 0
        for i, j in zip(output["Label"], ground_trth):
            if i == j:
                c += 1
        a = c/len(output)

        output.to_csv(f'Q1_{d}_GloVe_similarity_{t}.csv')
        with open(f'Q1_{d}_GloVe_similarity_{t}.csv', 'a') as f:
            try:
                f.write(f"\nAccuracy = {'%.2f' % a}")
            except:
                pass
            f.close()


# In[ ]:





# In[ ]:




