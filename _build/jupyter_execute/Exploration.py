#!/usr/bin/env python
# coding: utf-8

# # Data Exploration

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from functools import reduce


# Reading in the file

# In[2]:


os.listdir()


# In[3]:


data = pd.read_csv('L2 Data Scientist Assessment - Data.csv', dtype = str, encoding = 'cp1252')


# Quick look at what columns are in the dataset

# In[4]:


data.columns.values


# Check what classes there are

# In[5]:


data['class'].unique()


# Convert certain columns to numeric fields for calculations later on and use the sum of TP/FP/TN/FN to see how many total Pixels there are in the image dataset

# In[6]:


data['TP'] = data['TP'].astype(int)
data['FP'] = data['FP'].astype(int)
data['TN'] = data['TN'].astype(int)
data['FN'] = data['FN'].astype(int)

data['f1'] = data['f1'].astype(float)
data['accuracy'] = data['accuracy'].astype(float)

data['Check'] = data['TP'] + data['FP'] + data['TN'] + data['FN']
data['Check'].unique()


# Create separate dataframes for each class and check the size of these

# In[7]:


background = data[data['class'] == 'Background']
tissue = data[data['class'] == 'Tissue']
lesions = data[data['class'] == 'Lesions']

len(background), len(tissue), len(lesions)


# Creating some useful variables to describe the given dataset and also the image dataset that was classified 

# In[8]:


total_tissue_pixels = tissue['TP'].iloc[0] + tissue['FN'].iloc[0]
total_background_pixels = background['TP'].iloc[0] + background['FN'].iloc[0]
total_lesions_pixels = lesions['TP'].iloc[0] + lesions['FN'].iloc[0]
total_pixels = total_tissue_pixels + total_background_pixels + total_lesions_pixels
total_classifiers = len(data['classifier'].unique())
total_models = len(data['model'].unique())
total_experiments = len(data)


# Output some details about the given dataset and also derived details about the image dataset the experimentation was performed on

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


# Quick check to see how difficult the three classes are to classify, looks like Lesions are the trickiest which makes sense

# In[10]:


tissue_f1_mean = tissue['f1'].mean()
background_f1_mean = background['f1'].mean()
lesions_f1_mean = lesions['f1'].mean()

background_f1_mean, tissue_f1_mean, lesions_f1_mean

