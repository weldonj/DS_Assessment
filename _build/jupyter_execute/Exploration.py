#!/usr/bin/env python
# coding: utf-8

# # Data Exploration

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from functools import reduce


# In[2]:


os.listdir()


# In[3]:


data = pd.read_csv('L2 Data Scientist Assessment - Data.csv', dtype = str, encoding = 'cp1252')


# In[4]:


data.columns.values


# In[5]:


data['class'].unique()


# In[6]:


# check total number of pixels

data['TP'] = data['TP'].astype(int)
data['FP'] = data['FP'].astype(int)
data['TN'] = data['TN'].astype(int)
data['FN'] = data['FN'].astype(int)

data['f1'] = data['f1'].astype(float)
data['accuracy'] = data['accuracy'].astype(float)

data['Check'] = data['TP'] + data['FP'] + data['TN'] + data['FN']
data['Check'].unique()


# In[7]:


background = data[data['class'] == 'Background']
tissue = data[data['class'] == 'Tissue']
lesions = data[data['class'] == 'Lesions']

len(background), len(tissue), len(lesions)


# In[8]:


total_tissue_pixels = tissue['TP'].iloc[0] + tissue['FN'].iloc[0]
total_background_pixels = background['TP'].iloc[0] + background['FN'].iloc[0]
total_lesions_pixels = lesions['TP'].iloc[0] + lesions['FN'].iloc[0]
total_pixels = total_tissue_pixels + total_background_pixels + total_lesions_pixels
total_classifiers = len(data['classifier'].unique())
total_models = len(data['model'].unique())
total_experiments = len(data)


# In[9]:


print(f'''Image Dataset Contains {int(total_pixels)} total Pixels\n
Image Dataset Contains {int(total_pixels/(512*512))} Images if each Image is 512x512 in size
Image Dataset Contains {int(total_pixels/(256*256))} Images if each Image is 256x256 in size\n
{total_tissue_pixels/total_pixels*100:.2f}% of all Pixels are Tissue
{total_background_pixels/total_pixels*100:.2f}% of all Pixels are Background
{total_lesions_pixels/total_pixels*100:.2f}% of all Pixels are Lesions\n
Experiment Dataset Contains {total_experiments} total Experiments
Experiment Dataset Contains {total_classifiers} unique Classifiers
Experiment Dataset Contains {total_models} unique Models''')


# In[10]:


tissue_f1_mean = tissue['f1'].mean()
background_f1_mean = background['f1'].mean()
lesions_f1_mean = lesions['f1'].mean()

background_f1_mean, tissue_f1_mean, lesions_f1_mean


# In[ ]:




