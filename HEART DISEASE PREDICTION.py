#!/usr/bin/env python
# coding: utf-8

# # HEART DISEASE PREDICTION USING LOGISTIC REGRESSION

# In[10]:


import pandas as pd


# In[11]:


data=pd.read_csv("hdp.csv")


# In[12]:


data


# In[13]:


x=data[["age","cigsPerDay","totChol","sysBP","diaBP","BMI","heartRate","glucose"]]


# In[14]:


y=data[["TenYearCHD"]]


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.9,shuffle=True)


# In[17]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[18]:


from sklearn.linear_model import LogisticRegression


# In[19]:


model=LogisticRegression()


# In[20]:


model.fit(x_train,y_train)


# In[21]:


predictions=model.predict(x_test)


# In[22]:


from sklearn.metrics import accuracy_score


# In[23]:


accuracy_score(y_test,predictions)


# In[24]:


import numpy as np


# In[25]:


p=pd.DataFrame(np.c_[predictions,y_test],columns=["Predicted","Actual"])


# In[26]:


p


# In[ ]:





# In[ ]:




