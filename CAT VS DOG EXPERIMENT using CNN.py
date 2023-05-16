#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('pip install opencv-python')


# In[42]:


from glob import glob
from skimage.io import imread,imshow
import numpy as np
import cv2


# In[44]:


x=[]
y=[]


# In[46]:


for file in glob("C:\\Users\\DELL\\Documents\\train\\*.jpg"):
    if 'cat' in file.split('\\')[-1]:
        y.append(1)
    else:
        y.append(0)
    image=imread(file)
    image=cv2.resize(image,(224,224))
    x.append(image)


# In[47]:


import numpy as np
x=np.array(x)
y=np.array(y)


# In[48]:


x.shape,y.shape


# In[49]:


import random


# In[50]:


idx=random.sample(range(0,25000),5000)


# In[51]:


idx


# In[52]:


x_random=x[idx]
y_random=y[idx]


# In[54]:


x_random=x_random/x_random.max()


# In[55]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow((x_random[i]))
    plt.axis('off')


# In[53]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_random,y_random,test_size=0.2)


# In[76]:


from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense


# In[80]:


model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(1,activation="sigmoid"))


# In[81]:


model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])


# In[82]:


history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5,batch_size=64)


# In[ ]:




