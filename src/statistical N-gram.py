#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import io

import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


# In[48]:

#dataset
data = pd.read_excel('../dataset/processed_dataset.xlsx')   
#final_excel = pd.read_excel('./output_readability_lstm.xlsx','Baseline_scores')


# In[49]:


import nltk
import re
from collections import Counter
import sys 
import random
from nltk.util import ngrams
#nltk.download('punkt')
import math


# In[50]:


ngrams_c = {}
ngram = []
unigram = []
bigram = []
trigram = []


# In[51]:


def find_val(gram):
    if len(gram) == 0:
        return sum(ngrams_c[1].values())
    if gram in ngrams_c[len(gram)].keys():
        return ngrams_c[len(gram)][gram] 
    else:
        return 0


# In[52]:


def kneser(gram, maxi):
    x = len(gram)
    d = 0.75
    if find_val(gram[:-1]) == 0:
        lamda = random.uniform(0,1)
        term = random.uniform(0.00001,0.0001)
    else:
        if x == maxi:
            term = max(0,find_val(gram)-d)/find_val(gram[:-1])
        else:
            term = max(0,(sum(token[1:] == gram for token in ngrams_c[len(gram) + 1].keys()) - d)) / find_val(gram[:-1])
        lamda = d * sum(token[:-1] == gram[:-1] for token in ngrams_c[len(gram)].keys()) / find_val(gram[:-1])
    if x == 1:
       return term
    else:
        return term + lamda * kneser(gram[:-1], maxi)


# In[66]:


def calculate_p(data,n):
  for i in range(0,len(data['SENTENCES'])):
    words = nltk.word_tokenize(str(data['SENTENCES'][i]))
    for w in range(0,len(words)):
          unigram.append(tuple(words[w]))
  ngrams_c[1] = Counter(unigram)
  for i in range(0,len(data['SENTENCES'])):
    words = nltk.word_tokenize(str(data['SENTENCES'][i]))
    for w in range(0,len(words)-1):
            bigram.append(tuple(words[w:w+2]))
           
  ngrams_c[2] = Counter(bigram)
  for i in range(0,len(data['SENTENCES'])):
    words = nltk.word_tokenize(str(data['SENTENCES'][i]))
    for w in range(0,len(words)-2):
            trigram.append(tuple(words[w:w+3]))
  ngrams_c[3] = Counter(trigram)
  for i in range(0,len(data['SENTENCES'])):
    words = nltk.word_tokenize(str(data['SENTENCES'][i]))
    for w in range(0,len(words)-n+1):
            ngram.append(tuple(words[w:w+n]))
  ngrams_c[4] = Counter(ngram) 
  #print(ngrams_c)
  for c in range(0,500): #no of sentences to print results
    inp = str(data['SENTENCES'][c])
    
    words = nltk.word_tokenize(inp)
    tex = []
    ngram_inp = []
    #E2 model
    if c!=0:
        PSE = data['SENTENCES'][c-1][-1]
        tex.append(tuple([PSE,words[0]]))
        tex.append(tuple([PSE,words[0],words[1]]))
        tex.append(tuple([PSE,words[0],words[1],words[2]]))
    else:
        tex.append(tuple([words[0]]))
        tex.append(tuple([words[0],words[1]]))
        tex.append(tuple([words[0],words[1],words[2]]))
    for x in range(0,len(words)-n+1):
        tex.append(tuple(words[x:x+n]))
    #print(tex)
    ngram_inp = Counter(tex)
    #m=len(ngram_inp)
    p=1
    perp = 0
    M = 0
    word_list =inp.split()
    m=len(word_list)
    probability = []
    perplexity = []
    i=0
    for gram in ngram_inp:
            prob =  kneser(gram, len(gram))
            #print(prob)
            if prob!=0:
                probability.append(prob)
                p *= prob
                if p!=0:
                  perp=1/p
                if m>0:
                  perp=perp**(float(1/m))
                perplexity.append(perp)
#             if prob!=0:
#                 M+=math.log(prob,m)
        
    #E3
    f = 1/m
    perplexity.sort()
    # print(i)
    #E3
    if m!= 1:
      valgood = perplexity[int(len(perplexity)*0.6)]
      valbad = perplexity[int(len(perplexity)*0.4)]
      for i in range(len(perplexity)):
        w = perplexity[i]
        if w>=valgood:
          f=f*w
        if w<=valbad:
          f=f/w
    #print(p)
    #print('fluency by E1 = ',M/m)
    #print(inp)
    print(f)


# In[67]:


calculate_p(data,4)


# In[57]:



# In[ ]:




