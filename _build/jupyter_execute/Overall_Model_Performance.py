#!/usr/bin/env python
# coding: utf-8

# # Overall Model Performance

# In[71]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from functools import reduce
from scipy.stats.stats import pearsonr


# In[5]:


data = pd.read_csv('L2 Data Scientist Assessment - Data.csv', dtype = str, encoding = 'cp1252')


# In[6]:


data['f1'] = data['f1'].astype(float)
data['accuracy'] = data['accuracy'].astype(float)

background = data[data['class'] == 'Background']
tissue = data[data['class'] == 'Tissue']
lesions = data[data['class'] == 'Lesions']


# In[42]:


tissue_models_mean = tissue.groupby('model').agg({'f1':'mean'}).reset_index()
background_models_mean = background.groupby('model').agg({'f1':'mean'}).reset_index()
lesions_models_mean = lesions.groupby('model').agg({'f1':'mean'}).reset_index()


# In[43]:


def plot(objects,performance,title):
    y_pos = np.arange(len(objects))
    objects = [x for _,x in sorted(zip(performance,objects),reverse=True)]
    performance = [x for x,_ in sorted(zip(performance,objects),reverse=True)]
    plt.figure(figsize=(16, 10))
    plt.bar(y_pos, performance, align='center', alpha=0.5, width=0.7)
    plt.xticks(y_pos, objects)
    plt.ylabel('Average F1')
    plt.title(title)
    plt.ylim([(min(performance) - 0.1), 1])
    plt.xticks(rotation=45)
    plt.show()


# ## Background Pixel Classification

# In[44]:


plot(list(background_models_mean['model']),list(background_models_mean['f1']),'Background Classification Performance')


# AE_Xception has the highest average performance for Background Pixel Classification

# ## Tissue Pixel Classification

# In[45]:


plot(list(tissue_models_mean['model']),list(tissue_models_mean['f1']),'Tissue Classification Performance')


# Seg_Model has the highest average performance for Tissue Pixel Classification

# ## Lesion Pixel Classification

# In[8]:


plot(list(lesions_models_mean['model']),list(lesions_models_mean['f1']),'Lesions Classification Performance')


# Seg_Model also has the highest average performance for Lesion Pixel Classification

# Now to get the absolute top performers to ensure that there aren't some outlying excellnt results for other models

# In[41]:


tissue_absolute_best = tissue[['model','f1']].sort_values(by='f1',ascending=False).head(20)
lesions_absolute_best = lesions[['model','f1']].sort_values(by='f1',ascending=False).head(20)


# In[47]:


tissue_absolute_best


# In[48]:


lesions_absolute_best


# Looks good for both Tissue and Lesion, Seg_Model not only looks to be the highest average performer, but also makes up the top 20 absolute best performer for both Tissue and Lesion

# Now to see if there is a correlation between the model's performance on these two Pixel types

# In[85]:


ti = list(tissue[tissue['model'] == 'Seg_Model'].sort_values(by='classifier')['f1'].dropna())


# In[86]:


le = list(lesions[lesions['model'] == 'Seg_Model'].sort_values(by='classifier')['f1'].dropna())


# In[98]:


fig = plt.figure(figsize=(10.5,6))
fig.suptitle('Seg_Model Tissue and Lesion Correlation', fontsize=20)
plt.xlabel('Lesion Classification Performance')
plt.ylabel('Tissue Classification Performance')
plt.xlim(0.25,0.85)
plt.ylim(0.93,0.985)
plt.scatter(le, ti)


# In[87]:


correlation, p_value = pearsonr(le, ti)


# In[88]:


correlation


# In[99]:


p_value


# Correlation greater than 0.8 with a P-Value well below 0.05 indicates a true, strong, positive correlation, indicating that this is likely a good choice of model for both Tissue and Lesion Classification

# In[ ]:




