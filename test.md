Boosting Insights
================

Introduction
------------

The following will test boosting capabilities to predict incomes given
some sociodemographic variables. We will then compare boosting to other
Machine Learning methods and try to understand the differences.

Exploratory Data Analyis
------------------------

First of all, let us deep-dive in the dataset.

![](test_files/figure-markdown_github-ascii_identifiers/cars-1.png)

Boosting Review
---------------

Boosting classifies values based on a weighted vote:
![H(x)=sign(\\sum\_{t=1}^T\\alpha\_th\_t(x))](https://latex.codecogs.com/png.latex?H%28x%29%3Dsign%28%5Csum_%7Bt%3D1%7D%5ET%5Calpha_th_t%28x%29%29 "H(x)=sign(\sum_{t=1}^T\alpha_th_t(x))")
