#!/usr/bin/env python
# coding: utf-8

# # SALARY PREDICTION PROGRAM

# In[2]:


import pandas as pd


# In[3]:


data=pd.read_csv("salary.csv")


# In[4]:


data


# In[5]:


x=data[["years of experience"]]


# In[6]:


y=data[["salary"]]


# In[7]:


x.shape,y.shape


# In[9]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True)


# In[14]:


(x_train.shape,y_train.shape),(x_test.shape,y_test.shape)


# In[15]:


from sklearn.linear_model import LinearRegression


# In[17]:


model=LinearRegression()


# In[18]:


model.fit(x_train,y_train)


# In[25]:


a=model.predict([[1.5]]).squeeze()


# In[26]:


print(a)


# In[27]:


predictions=model.predict(x_test)


# to evaluate predictions we import r2 score

# In[28]:


from sklearn.metrics import r2_score


# In[29]:


r2_score(predictions,y_test)


# In[38]:


aa=pd.DataFrame(np.c_[y_test,predictions],columns=["Actual","Predictions"])


# In[39]:


aa

