#!/usr/bin/env python
# coding: utf-8

# # Parameter Investigation

# In[3]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from functools import reduce
data = pd.read_csv('L2 Data Scientist Assessment - Data.csv', dtype = str, encoding = 'cp1252')

data['f1'] = data['f1'].astype(float)
data['accuracy'] = data['accuracy'].astype(float)

background = data[data['class'] == 'Background']
tissue = data[data['class'] == 'Tissue']
lesions = data[data['class'] == 'Lesions']


# In[25]:


parameters = ['dropoutFraction', 'augmentColor', 'augmentGeometry', 'balanceClasses', 'elasticDeform']
dfs = [background, tissue, lesions]


# In[26]:


def f1_group(df, param):
    return df.groupby(['model',param]).agg({'f1':'mean'}).reset_index().sort_values(by=['f1'], ascending=False)


# In[28]:


for param in parameters:
    print(f'Effect of "{param}" on Lesion Classification\n'.split('=')[0])
    print(f1_group(lesions, param))
    print()


# In[29]:


for param in parameters:
    print(f'Effect of "{param}" on Tissue Classification\n'.split('=')[0])
    print(f1_group(tissue, param))
    print()


# In[ ]:




