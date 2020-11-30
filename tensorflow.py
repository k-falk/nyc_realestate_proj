# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 19:34:40 2020

@author: Xkfal
"""

import keras 
import numpy as np
import pandas as pd

path = 'C:/Users/Xkfal/Documents/nyc_realestate_proj/'
df = pd.read_csv(path + 'explored_data.csv')
df.head()

df_model = df[['sale_price','borough', 'building_category', 'neighborhood', 'tax_class', 'total_units', 'distance', 'square_ft', 'age']]
avgna = df.loc[df['total_units'].isna(), 'sale_price'].median()


#df_model['comm_units'] = df_model['comm_units'].fillna(value = 0)
df_model.isnull().mean()
#df_model = df_model[~df_model.isin([np.nan, np.inf, -np.inf]).any(1)]
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df_model[[ 'distance', 'square_ft', 'age']]=imputer.fit(df_model[['distance', 'square_ft', 'age']].values)
df_model.isnull().mean()

from sklearn.model_selection import train_test_split


df_dumb = pd.get_dummies(df_model)
X = df_dumb.drop('sale_price', axis=1)
y = df_dumb.sale_price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import datetime
import os


##First thing we do is define the model:
##Our model is going to use relu activation and adam optimizer with mean absolute error as our loss minimizer
# define base model

model = keras.Sequential()
model.add(keras.layers.Dense(10, activation = 'relu', input_shape =(301,)))
model.add(keras.layers.Dense(10, activation = 'relu'))
model.add(keras.layers.Dense(1,activation = None))
#####
## We tested deeper models and wider models but this basic model worked great and was fast without compromising performance


model.compile(optimizer="adam", loss="mean_absolute_error")
model.fit(X_train, y_train, epochs=30)
model.evaluate(X_test,y_test)

model.save(path + '/model')
loaded_model = tf.keras.models.load_model(path + '/model')
from data_input import data_in
x = np.array(data_in).reshape(1,-1)
model.predict(x)[0][0]

predictions = np.squeeze((model.predict(X_test)))
y = np.array((y_test),ndmin=1)
mae = np.mean(abs(np.expm1(y) - np.expm1(predictions)))
print('MAE in Dollars:', mae)

import matplotlib.pyplot as plt
plt.scatter(np.expm1(y_test),np.expm1(predictions), color = "red")
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)

list_ = X_test.iloc[0,:]
list(zip(*list_))

model.predict(list(X_test.iloc[0,:]))

# custom R2-score metrics for keras backend
from keras import backend as K

# coefficient of determination (R^2) for regression  (only for Keras tensors)
#def r_square(y_true, y_pred):
#    from keras import backend as K
#    SS_res =  K.sum(K.square(y_true - y_pred)) 
#    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
#    return ( 1 - SS_res/(SS_tot + K.epsilon()) )




#
#coeff_determination(y_test, predictions)
#
#plt.scatter(np.expm1(y_test),np.expm1(predictions), color = "red")
#plt.xlabel('Actual values')
#plt.ylabel('Predicted values')
#xpoints = ypoints = plt.xlim()
#plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
#
#error = predictions - y_test
#plt.hist(error, bins=25)
#plt.xlabel('Prediction Error [Price]')
#_ = plt.ylabel('Count')
#
#
#
#def make_square_axes(ax):
#    """Make an axes square in screen units.
#
#    Should be called after plotting.
#    """
#    ax.set_aspect(1 / ax.get_data_ratio())
#
#plt.scatter(y_test,predictions)
#xpoints = ypoints = plt.xlim()
#plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
#
#
#for i, y in enumerate(y_test.head()):
#    delta  = predictions[i] -  np.expm1(y)
#    print('Delta:' , delta )
#    
