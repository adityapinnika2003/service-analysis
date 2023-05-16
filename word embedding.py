#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import imdb


# In[3]:


(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=10000)


# In[5]:


word_index=imdb.get_word_index()


# In[7]:


#padding for same length
from keras.utils import pad_sequences
x_train_seq=pad_sequences(x_train,200,padding='post')


# In[8]:


x_test_seq=pad_sequences(x_test,200,padding='post')


# In[11]:


from keras.models import Sequential
from keras.layers import Dense,Embedding,Flatten


# In[22]:


model=Sequential()
model.add(Embedding(input_dim=10000,output_dim=8,input_length=200))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(1,activation="sigmoid"))


# In[23]:


model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])


# In[24]:


model.fit(x_train_seq,y_train,validation_data=(x_test_seq,y_test),epochs=10)


# In[ ]:




