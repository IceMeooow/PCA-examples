# PCA examples

This project is about Principal Component Analysis. 
PCA is a transformation that attempts to convert possibly correlated features into a set of linearly uncorrelated ones.

### Data

[UCI's Chronic Kidney Disease](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease) data set are used. It is a collection of samples taken from patients in India over a two month period, some of whom were in the early stages of the disease.


### Technology


* python: 3.6.1

* pandas: 0.20.1

* matplotlib: 2.0.2

* scikit-learn: 0.19



### Description 

This project contain two py files:
* kidney_disease.py
* kidney_disease(more_than_3_features).py

***Kidney_disease.py file:***
A subset of kidney disease data is used. It contains only three columns (bgr, rc, and wc).  PCA is applied to it for reduction from 3D to 2D space. This file contains 3D visualization of subset and  visualization of data after PCA. The PCA's graphs show how data is located without  scaling features and when it is done.

***Kidney_disease(more_than_3_features).py file:***
Two subsets of kidney disease data are used. One of them contains only numerical features, another - numerical and nominal features.
PCA is applied to it for reduction from nD to 2D space. There are visualization of data after PCA. The PCA's graphs show how data is located without scaling features and when it is done.







### Getting starded


* Download this file or copy repository.


* Open file in your IDE for PYTHON.


* Run file.



 

### Help

* About [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis).
* [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) in scikit - learn library.
