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

# # TensorFlow Linear Regression

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

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ## **The Auto MPG dataset**
#
# The dataset is available from the UCI Machine Learning Repository

# +
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model_Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
# -

dataset = raw_dataset.copy()
dataset.tail()

# ## **Clean the data**

#Unkown values in the dataset:
dataset.isna().sum()

#Drop na rows
dataset = dataset.dropna()

#The "Origin" column is categorical, not numeric. So the next step is to one-hot encode the values in the column with pd.get_dummies
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()

# ## **Split the data into training and test sets**
# Test set will be used in the final evaluation of the models

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# ## **Inspect the data**
# Review the joint distribution of a few pairs of columns from the training set.
#
# The top row suggests that the fuel efficiency (MPG) is a function of all the other parameters. The other rows indicate they are functions of each other.

sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

#Check summary stats. Note how each feature covers a very different range
train_dataset.describe().transpose()

# ## **Split features from labels**
# Separate the target value--the "label"--from the features. This label is the value that you will train the model to predict.

# +
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')
# -

# ## **Normalization**
# Due to the ranges of each feature being very different, the data needs to be normalized.
#
# It is good practice to normalize features that use different scales and ranges.
#
# One reason this is important is because the features are multiplied by the model weights. So, the scale of the outputs and the scale of the gradients are affected by the scale of the inputs.
#
# Although a model might converge without feature normalization, normalization makes training much more stable.

# ## **The Normalization Layer**

#Normalized layer
normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

print(normalizer.mean.numpy())

# +
first = np.array(train_features[:1])

with np.printoptions(precision=4, suppress=True):
    print("First example:", first)
    print("Normalized:", normalizer(first).numpy())
# -

# ## **Linear Regression**
# Linear regression with one variable
#
# Begin with a single-variable linear regression to predict 'MPG' from 'Horsepower'.
#
# Training a model with tf.keras typically starts by defining the model architecture. Use a tf.keras.Sequential model, which represents a sequence of steps.
#
# There are two steps in your single-variable linear regression model:
#
# Normalize the 'Horsepower' input features using the tf.keras.layers.Normalization preprocessing layer.
#
# Apply a linear transformation (y=mx+b) to produce 1 output using a linear layer (tf.keras.layers.Dense).
#
# The number of inputs can either be set by the input_shape argument, or automatically when the model is run for the first time.

# +
#Create a NumPy array made of the 'Horsepower' features. Then normalize and fit its state to the horsepower data
horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = layers.experimental.preprocessing.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

# +
#Keras Sequential model:
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

horsepower_model.summary()
# -

# **This model will predict 'MPG' from 'Horsepower'**
#
# Run the untrained model on the first 10 'Horsepower' values. The output won't be good, but notice that it has the expected shape of (10,1)

horsepower_model.predict(horsepower[:10])

# Once the model is built, configure the training procedure using the Keras Model.compile method. The most important arguments to compile are the loss and the optimizer, since these define what will be optimized (mean_absolute_error) and how (using the tf.keras.optimizers.Adam).

horsepower_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

# Having configured the training, use Keras Model.fit to execute the training for 100 epochs
# %%time
history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split=0.2
)

# Visualize the model's training progress using the stats stored in the 'history' object
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# +
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0,10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

plot_loss(history)
# -


