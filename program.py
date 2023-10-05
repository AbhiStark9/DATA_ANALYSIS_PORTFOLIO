#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[8]:


data = pd.read_csv("housing.csv")


# In[9]:


data


# In[10]:


data.info()


# In[11]:


data.dropna(inplace = True)


# In[12]:


data.info()


# In[17]:


x = data.drop(["median_house_value"], axis=1)
y = data["median_house_value"]


# In[21]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[22]:


train_data = x_train.join(y_train)


# In[23]:


train_data


# In[25]:


train_data.hist(figsize=(15,8))


# In[26]:


plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap='YlGnBu')


# In[27]:


train_data["total_rooms"]= np.log(train_data["total_rooms"]+1)
train_data["total_bedrooms"]= np.log(train_data["total_bedrooms"]+1)
train_data["population"]= np.log(train_data["population"]+1)
train_data["households"]= np.log(train_data["households"]+1)


# In[28]:


train_data.hist(figsize=(15,8))


# In[29]:


train_data.ocean_proximity.value_counts()


# In[34]:


train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(["ocean_proximity"], axis=1)


# In[35]:


train_data


# In[36]:


plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap='YlGnBu')


# In[38]:


plt.figure(figsize=(15,8))
sns.scatterplot(x="latitude", y="longitude", data=train_data, hue="median_house_value", palette="coolwarm")


# In[40]:


train_data["bedroom_ratio"] = train_data["total_bedrooms"]/train_data["total_rooms"]
train_data["houshold_rooms"] = train_data["total_rooms"]/train_data["households"]


# In[41]:


plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap='YlGnBu')


# # Linear Regression

# In[55]:


from sklearn.linear_model import LinearRegression


x_train, y_train = train_data.drop(["median_house_value"], axis=1),train_data["median_house_value"]


reg = LinearRegression()

reg.fit(x_train, y_train)


# In[49]:


test_data = x_test.join(y_test)

test_data["total_rooms"]= np.log(test_data["total_rooms"]+1)
test_data["total_bedrooms"]= np.log(test_data["total_bedrooms"]+1)
test_data["population"]= np.log(test_data["population"]+1)
test_data["households"]= np.log(test_data["households"]+1)

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(["ocean_proximity"], axis=1)

test_data["bedroom_ratio"] = test_data["total_bedrooms"]/test_data["total_rooms"]
test_data["houshold_rooms"] = test_data["total_rooms"]/test_data["households"]


# In[ ]:


x_test, y_test = test_data.drop(["median_house_value"], axis=1),test_data["median_house_value"]


# In[50]:


reg.score(x_test, y_test)


# In[58]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(x_train, y_train)


# In[59]:


forest.score(x_test, y_test)


# In[ ]:




