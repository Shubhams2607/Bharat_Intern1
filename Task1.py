#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


iris=pd.read_csv("IRIS.csv")


# In[3]:


iris.head(10)


# In[4]:


iris.info()


# In[5]:


iris.describe()


# In[6]:


iris


# In[7]:


iris.shape


# In[8]:


fig1=sns.pairplot(iris)
fig1.fig.set_figheight(6)
fig1.fig.set_figwidth(10)
plt.show()


# In[9]:


sns.FacetGrid(iris, hue="species",height=6).map(plt.scatter, "petal_length", "sepal_width").add_legend()
plt.show()


# In[10]:


X=iris.iloc[:,0:4]
Y=iris.iloc[:,4]
X.shape,Y.shape


# In[11]:


flower_mapping = {'setosa': 0,'versicolor': 1,'virginica':2}
iris["Species"] = iris["species"].map(flower_mapping)


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y)


# In[15]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)


# In[16]:


prediction = model.predict(x_test)


# In[17]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction)*100)


# In[ ]:




