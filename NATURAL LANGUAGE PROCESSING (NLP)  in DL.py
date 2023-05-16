#!/usr/bin/env python
# coding: utf-8

# In[38]:


text=["I am Aditya","&","This is DL class ","!"]


# # 1.TOKENIZATION

# In[39]:


voc=[]
for i in text:
    for j in i.split():
        voc.append(j)


# In[40]:


voc


# # 2.Conversion to Lowercase

# In[41]:


voc1=[]
for i  in voc:
    voc1.append(i.lower())


# In[42]:


voc1


# # 3.Removing Punctuation

# In[43]:


import string


# In[44]:


p=list(string.punctuation)


# In[45]:


p


# In[46]:


voc2=[]
for i in voc1:
    if i not in p:
        voc2.append(i)


# In[47]:


voc2


# # 4.Removing Stopwords

# In[52]:


nltk.download("stopwords")


# In[53]:


import nltk


# In[54]:


from nltk.corpus import stopwords


# In[55]:


sw=stopwords.words('english')


# In[56]:


sw


# In[57]:


voc3=[]
for  i in voc2:
    if i not in sw:
        voc3.append(i)


# In[58]:


voc3


# # 5.Stemming

# In[63]:


from nltk.stem import PorterStemmer


# In[64]:


stemmer=PorterStemmer()


# In[66]:


[stemmer.stem(w.lower()) for w in ["talk","talks","talking"]]


# 
