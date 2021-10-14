# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: 'Python 3.9.7 64-bit (''tf'': conda)'
#     name: python3
# ---

# # TensorFlow Regression

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read
np.set_printoptions(precision=3, suppress=True)

# +
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print("tenserflow version:", tf.__version__)
# -

# ## The Auto MPG dataset
#
# The dataset is available from the UCI Machine Learning Repository

# +
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model_Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
# -

dataset = raw_dataset.copy()
dataset.tail()

# ## Clean the data

#Unkown values in the dataset:
dataset.isna().sum()

#Drop na rows
dataset = dataset.dropna()

#The "Origin" column is categorical, not numeric. So the next step is to one-hot encode the values in the column with pd.get_dummies
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()

# ## Split the data into training and test sets
# Test set will be used in the final evaluation of the models

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# ## Inspect the data
# Review the joint distribution of a few pairs of columns from the training set.
#
# The top row suggests that the fuel efficiency (MPG) is a function of all the other parameters. The other rows indicate they are functions of each other.

sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

#Check summary stats. Note how each feature covers a very different range
train_dataset.describe().transpose()


