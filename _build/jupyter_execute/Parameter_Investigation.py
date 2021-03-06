#!/usr/bin/env python
# coding: utf-8

# # Parameter Investigation

# In[2]:


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


# Store a list of the model parameters

# In[3]:


parameters = ['dropoutFraction', 'augmentColor', 'augmentGeometry', 'balanceClasses', 'elasticDeform']
dfs = [background, tissue, lesions]


# Function to group by Model and Parameter so we can compare the impact on f1 score

# In[4]:


def f1_group(df, param):
    return df.groupby(['model',param]).agg({'f1':'mean'}).reset_index().sort_values(by=['f1'], ascending=False)


# Loop through the parameters and show which model + parameter combinations scored highest

# # Parameter effect on Lesion Classification

# In[8]:


for param in parameters:
    print(f'Effect of "{param}" on Lesion Classification\n')
    print(f1_group(lesions, param))
    print()


# We can see that for "dropoutFraction", the models that have both 0.0 and 0.2 all performed better with a dropoutFraction of 0
# 
# "augmentColor" set to FALSE with the Seg_Model model gives the overall best result, but for some other model TRUE outperforms FALSE
# 
# "augmentGeometry" provides a significant improvement for the top performing models when set to TRUE
# 
# Likewise, "elasticDeform" provides a signifcant improvement for the top performing models when set to TRUE        
# 
# 

# # Parameter effect on Tissue Classification

# In[9]:


for param in parameters:
    print(f'Effect of "{param}" on Tissue Classification\n')
    print(f1_group(tissue, param))
    print()


# The Optimal choices for all Parameters are the same for Tissue Classification as they were for Lesion Classification - This is a good outcome for us as it likely means we can use the same Classifier for both Lesion and Tissue Classification

# # Parameter effect on Background Classification

# In[10]:


for param in parameters:
    print(f'Effect of "{param}" on Tissue Classification\n')
    print(f1_group(background, param))
    print()


# Background Classification appears to be best with a different model - AE_Xception
# This should have a dropoutFraction of 0.2, augmentColor set to TRUE, augmentGeometry set to FALSE, balanceClasses set to TRUE and elasticDeform set to TRUE

# In[ ]:




