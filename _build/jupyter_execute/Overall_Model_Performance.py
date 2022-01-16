#!/usr/bin/env python
# coding: utf-8

# # Overall Model Performance

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from functools import reduce


# In[2]:


data = pd.read_csv('L2 Data Scientist Assessment - Data.csv', dtype = str, encoding = 'cp1252')


# In[3]:


data['f1'] = data['f1'].astype(float)
data['accuracy'] = data['accuracy'].astype(float)

background = data[data['class'] == 'Background']
tissue = data[data['class'] == 'Tissue']
lesions = data[data['class'] == 'Lesions']


# In[4]:


tissue_models_mean = tissue.groupby('model').agg({'f1':'mean'}).reset_index()
background_models_mean = background.groupby('model').agg({'f1':'mean'}).reset_index()
lesions_models_mean = lesions.groupby('model').agg({'f1':'mean'}).reset_index()


# In[5]:


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


# In[7]:


plot(list(background_models_mean['model']),list(background_models_mean['f1']),'Background Classification Performance')


# In[6]:


plot(list(tissue_models_mean['model']),list(tissue_models_mean['f1']),'Tissue Classification Performance')


# In[8]:


plot(list(lesions_models_mean['model']),list(lesions_models_mean['f1']),'Lesions Classification Performance')


# In[ ]:




