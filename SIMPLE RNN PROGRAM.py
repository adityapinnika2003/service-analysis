#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import imdb


# In[2]:


(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=10000)


# In[3]:


from keras.utils import pad_sequences
x_train_seq=pad_sequences(x_train,200,padding="post")
x_test_seq=pad_sequences(x_test,200,padding="post")


# In[4]:


x_train_seq[100]


# In[5]:


from keras.models import Sequential
from keras.layers import Embedding,Flatten,SimpleRNN,Dense,LSTM


# In[6]:


model=Sequential()
model.add(Embedding(input_dim=10000,output_dim=8,input_length=200))
#model.add(SimpleRNN(128))
model.add(LSTM(64))
model.add(Dense(1,activation="sigmoid"))


# In[7]:


model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])


# In[9]:


model.fit(x_train_seq,y_train,validation_data=(x_test_seq,y_test),epochs=1)


# In[ ]:




