#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv(r"C:\Users\PC\Downloads\archive (9)\kc_house_data.csv")


# In[3]:


df.head()


# In[4]:


df['date'] = pd.to_datetime(df['date'])


# In[5]:


df.head()


# In[6]:


# drop some columns that are not needed
df = df.drop(columns=['zipcode', 'lat', 'long'])


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


#create the scatter plot
plt.figure(figsize=(14, 10))
plt.scatter(df['price'], df['sqft_living'], c='black', label='House Rent')

#Add title
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.legend()
plt.show()


# In[11]:


df.corr()


# In[12]:


plt.hist(df['bedrooms'],bins=30)
plt.title('Distribution of bedrooms')
plt.show()


# In[13]:


# lable our x value
X = df.drop(columns=['price', 'date', 'id'])
X


# In[14]:


y = df['price']


# In[15]:


#Introduce train test from sklearn model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 


# In[16]:


#liner regression model 
lr = LinearRegression()


# In[17]:


lr.fit(X_train, y_train)


# In[18]:


C = lr.intercept_


# In[19]:


C


# In[20]:


p = lr.coef_
p


# In[21]:


#predit the price using prediting model set (training data)
y_pred_train = lr.predict(X_train)
y_pred_train


# In[22]:


#checking our prediction (training data)
import matplotlib.pyplot as plt
plt.scatter(y_train, y_pred_train)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.show()


# In[23]:


# Ascertain how good our model to the regression 
from sklearn.metrics import r2_score
r2_score(y_train, y_pred_train)


# In[24]:


y_pred_test = lr.predict(X_test)


# In[25]:


#checking our prediction (testing data)
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred_test)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.show()


# In[26]:


# Ascertain how good our model to the regression (testing data)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred_test)


# In[28]:


print('Downloading This File')


# In[ ]:




