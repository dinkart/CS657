#!/usr/bin/env python
# coding: utf-8

# In[1]:


import importlib
import csv
import json
import sys


# In[2]:


BR = importlib.import_module("Q2a")
TFIDF = importlib.import_module("Q2b")
BM25m = importlib.import_module("Q2c")


# In[3]:


fname = sys.argv[1]
query = {}
a_file = open(fname, encoding = "UTF-8")
next(a_file)
for line in a_file:
    line = line.split("\t")
    query[line[0]] = line[1]
a_file.close()


# In[4]:


w_coll = json.load(open('data.json'))
dic = {"QueryId" : [], "Iteration" : [], "DocId": [],"Relevance": []}


# In[ ]:


#Boolean Model
for qID in query:
    docs = BR.BRSystem(query[qID], w_coll)
    for i in range(len(docs)):
        dic["QueryId"].append(qID)
        dic["Iteration"].append(1)
        dic["Relevance"].append(1)
        dic["DocId"].append(docs[i])


# In[ ]:


f = open('Boolean.txt', 'w')
f.write("QueryId, Iteration, DocId, Relevance\n")
for i in range(len(dic["QueryId"])):
    f.write(f'{dic["QueryId"][i]}, {dic["Iteration"][i]}, {dic["DocId"][i]}.txt, {dic["Relevance"][i]}')
    f.write("\n")
f.close()
dic = {"QueryId" : [], "Iteration" : [], "DocId": [],"Relevance": []}


# In[ ]:


#Tf-Idf Model
for qID in query:
    docs = TFIDF.tfidf(query[qID], w_coll)
    for i in range(5):
        dic["QueryId"].append(qID)
        dic["Iteration"].append(1)
        dic["Relevance"].append(1)
        dic["DocId"].append(docs[i])


# In[ ]:


f = open('TFIDF.txt', 'w')
f.write("QueryId, Iteration, DocId, Relevance\n")
for i in range(len(dic["QueryId"])):
    f.write(f'{dic["QueryId"][i]}, {dic["Iteration"][i]}, {dic["DocId"][i]}.txt, {dic["Relevance"][i]}')
    f.write("\n")
f.close()
dic = {"QueryId" : [], "Iteration" : [], "DocId": [],"Relevance": []}


# In[5]:


#BM25 Model
for qID in query:
    docs = BM25m.BM25(query[qID], w_coll)
    for i in range(5):
        dic["QueryId"].append(qID)
        dic["Iteration"].append(1)
        dic["Relevance"].append(1)
        dic["DocId"].append(docs[i])


# In[6]:


f = open('BM25.txt', 'w')
f.write("QueryId, Iteration, DocId, Relevance\n")
for i in range(len(dic["QueryId"])):
    f.write(f'{dic["QueryId"][i]}, {dic["Iteration"][i]}, {dic["DocId"][i]}.txt, {dic["Relevance"][i]}')
    f.write("\n")
f.close()


# In[ ]:




