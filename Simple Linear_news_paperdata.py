#!/usr/bin/env python
# coding: utf-8

# # Import Data Set

# In[1]:


import pandas as pd
data = pd.read_csv("NewspaperData.csv")


# In[3]:


data


# In[4]:


data.info()


# # Correlation

# In[5]:


data.corr()


# In[7]:


import seaborn as sns
sns.distplot(data['daily'])


# In[8]:


import seaborn as sns
sns.distplot(data['sunday'])


# Fitting a Linear Regression Model

# In[9]:


import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data).fit()


# In[10]:


sns.regplot(x="daily", y="sunday", data=data);


# In[11]:


#Coefficients
model.params


# In[28]:


#t and p-Values
print(model.tvalues, '\n', model.pvalues)    


# In[13]:


#R squared values
(model.rsquared,model.rsquared_adj)


# # Predict for new data point

# In[22]:


#Predict for 200 and 300 daily circulation
newdata=pd.Series([200,300])


# In[23]:


newdata


# In[24]:


data_pred=pd.DataFrame(newdata,columns=['daily'])


# In[25]:


data_pred


# In[26]:


model.predict(data_pred)


# In[21]:


13.83+(1.34*200)


# In[ ]:




