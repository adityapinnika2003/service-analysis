#!/usr/bin/env python
# coding: utf-8

# In[18]:


text=["Dasara movie is super hit","Dharani character is good","vennela looks beautiful"]


# In[19]:


#preparing vocabulary
voc=[]
for i in text:
    for j in i.split():
        voc.append(j)


# In[20]:


voc


# In[21]:


len(voc)


# In[22]:


voc=set(voc)
voc=list(voc)


# In[23]:


voc


# In[24]:


len(voc)


# In[25]:


word_index={}


# In[26]:


#preparing dictionary
for i,word in enumerate(voc):
    word_index[word]=i


# In[27]:


word_index


# In[ ]:


#preparing text sequences


# In[28]:


sequences=[]
for sample in text:
    sequence=[]
    for i in sample.split():
        sequence.append(word_index.get(i))
    sequences.append(sequence)
    


# In[29]:


sequences


# In[31]:


#preparation of one hot encoding
import numpy as np
one_hot_encoding=np.zeros((len(text),len(voc)))
for i,sequence in enumerate(sequences):
    print(i,sequence)
    one_hot_encoding[i,sequence]=1


# In[32]:


one_hot_encoding


# In[ ]:


# using keras


# In[33]:


from keras.preprocessing.text import Tokenizer
token=Tokenizer(num_words=10000)


# In[34]:


token.fit_on_texts(text)


# In[35]:


sequences=token.texts_to_sequences(text)


# In[36]:


sequences


# In[ ]:




