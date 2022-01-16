# Data Scientist Assessment for Deciphex - John Weldon


This Jupyter Book documents the analysis I performed based on the instructions provided below:

:::{note}
Analysis of Experimental Results\
Assignment\
\
The dataset provided is a collection of results and classifier parameters from experiments that were run on a single dataset. The dataset has three classes: Background, Tissue and Lesion. The goal for the classifiers was to correctly detect pixels in an image that belonged to these 3 classes, with a focus on Lesion.\
\
Your task is to compare the results of these classifiers in the dataset and to visualise information that could give some insight into the effect of model parameters on the performance metrics.\
\
To be submitted\
\
A presentation of 5-6 slides needs to be created to summarise the task, findings of the data analysis and future suggestions based on the insights found.\
\
Code used to do the analysis\
\
Need to demonstrate:\
\
Understanding of the evaluation metrics and their relationship to confusion matrix (TP, FP, FN, TN) values.\
Ability to identify, summarise and clean-up (if needed) the outliers in the data\
Ability to perform statistical analysis on provided data \
Ability to visualise results from various angles and to explain how to interpret the visualisations\
Ability to clearly document the findings of the analysis and generate a report\
Ability to develop reusable solutions for data analysis\
\
Details\
Notes on the data in the provided CSV file (comparison_of_classifiers.csv).\
The classifiers that are investigated are all trained on a dataset called Heart_Lesions_10x_Static200618_Cons3 dataset (anno_set).\
The dataset has three classes, and the main focus should be on exploring the results of detecting the Lesions class.\
Correlation between results on the Tissue and Lesion classes is something of interest - the primary endpoint of these classifiers is to detect abnormal lesions in normal tissue.\
Binary class in this example is exactly the same as Lesions class, therefore can  be ignored.\
Each experiment is represented by a unique classifier name.\
The results are represented using metrics such as Precision, Sensitivity, F1 score, MCC, DOR, TP, FP, FN, TN.\
Parameters that are varied for the experiments are: model, dropoutFraction, AugmentColor, augmentGeo, balanceClasses, elasticDeformation. It is important to identify what are the effects on the results when those parameters are varied.
:::