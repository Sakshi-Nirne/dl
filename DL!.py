#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow import keras 
from tensorflow.keras import layers


# In[2]:


df=pd.read_csv(r"C:\Users\91895\Downloads\Dl\housing_data.csv")
df.head()


# In[3]:


df.isnull().sum()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.fillna(df.mean(), inplace=True)


# In[7]:


sns.distplot(df.MEDV)


# In[8]:


sns.boxplot(df.MEDV)


# In[9]:


corr=df.corr()
corr.loc['MEDV']


# In[10]:


sns.heatmap(corr, annot=True)


# In[11]:


features = ['LSTAT', 'RM', 'PTRATIO']

for i, col in enumerate(features):
    plt.subplot(1, len(features),(i+1))
    x = df[col]
    y = df.MEDV
    plt.scatter(x, y)
    plt.xlabel(col)
    plt.ylabel("House prices")


# In[12]:


X = df.iloc[:,:-1]
y= df.MEDV


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[14]:


mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# In[15]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[17]:


y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
print(rmse)


# In[18]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)


# In[19]:


import keras
from keras.layers import Dense
from keras.models import Sequential


# In[20]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[21]:


model = Sequential()
model.add(Dense(128, activation='relu', input_dim=13))


# In[22]:


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


# In[23]:


history = model.fit(X_train, y_train, epochs=100, validation_split=0.05)

# Plotting the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')


# In[25]:


from sklearn.metrics import mean_absolute_error
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
print('Mean squared error on test data: ', mse_lr)
print('Mean absolute error on test data: ', mae_lr)


# In[ ]:




