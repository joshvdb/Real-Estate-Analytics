# Real-Estate-Analytics

## Overview

A program that uses a variety of Statistical and Machine Learning models to predict the market value of property in Switzerland (for both purchase and rent), based on scraped data from property websites. Specifically, property attributes are scraped from the Comparis website (primarily using the **requests** package as it proves to be faster than **selenium**), and saved as a pandas DataFrame. The program saves the unique ID associated with each property listing, thus allowing the user to add to previously collected data with each re-run of the Jupyter notebook (duplicate entries are deleted). The data is then processed (e.g. missing value imputation, outlier detection and removal, scaling of continuous features) and fed into a model training pipeline that uses Bayesian hyperparameter optimization and k-fold cross-validation to find the optimal model (from a set of linear regression and tree-based models). Model feature importances are calculated, using the regression coefficients (in the case of the linear regression models) and TreeSHAP (in the case of tree-based models). The models and associated data is then saved.

A user can then load the models and associated data (shown in the section at the end of the Jupyter notebook: **Predict the price of any given property**), and use them to predict the market price of a given property. The Jupyter notebook also includes a  Price vs Living Space graph that plots the property price alongside properties that are in it's peer group (i.e. properties that have the same number of rooms and property type).

## Installation

The requirements file can be installed using the below commands in a terminal:

**Conda:** conda install --file requirements.txt

**Pip:** pip install -r requirements.txt

## TODO

1. Add time-series analysis tools to the program, such as a date stamp for property listings, and time-series graphs, to allow a user to analyse the change in property prices with time. While the Swiss property market is generally stable, this tool could prove useful during periods of high volatility, or when applied to the property markets of other countries.
