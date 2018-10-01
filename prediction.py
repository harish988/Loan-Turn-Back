# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 14:32:37 2018

@author: Siva
"""

import numpy as np
import pandas as pd

#importing training data
dataset = pd.read_csv("credit_train.csv")
dataset['Months since last delinquent'].fillna(0,inplace=True)
X_train = dataset.iloc[:,3:18].values
y_train = dataset.iloc[:,18].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
label_encoder_exp = LabelEncoder()
X_train[:, 4] = label_encoder_exp.fit_transform(X_train[:, 4].astype(str))
label_encoder_term = LabelEncoder()
X_train[:, 1] = label_encoder_term.fit_transform(X_train[:, 1].astype(str))
label_encoder_purpose = LabelEncoder()
X_train[:, 6] = label_encoder_purpose.fit_transform(X_train[:, 6].astype(str))
label_encoder_home_ownership = LabelEncoder()
X_train[:, 5] = label_encoder_home_ownership.fit_transform(X_train[:, 5].astype(str))

#Handling missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, 0:1])
X_train[:, 0:1] = imputer.transform(X_train[:, 0:1])

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_train[:, 1:2])
X_train[:, 1:2] = imputer.transform(X_train[:, 1:2])

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, 2:4])
X_train[:, 2:4] = imputer.transform(X_train[:, 2:4])

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_train[:, 4:7])
X_train[:, 4:7] = imputer.transform(X_train[:, 4:7])

imputer_bankruptcy = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer_bankruptcy = imputer_bankruptcy.fit(X_train[:, 14:15])
X_train[:, 14:15] = imputer_bankruptcy.transform(X_train[:, 14:15])

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_train[:, 7:8])
X_train[:, 7:8] = imputer.transform(X_train[:, 7:8])

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_train[:, 8:9])
X_train[:, 8:9] = imputer.transform(X_train[:, 8:9])

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_train[:, 10:12])
X_train[:, 10:12] = imputer.transform(X_train[:, 10:12])

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, 12:14])
X_train[:, 12:14] = imputer.transform(X_train[:, 12:14])

y_train = y_train.reshape(-1,1)
imputer_y_train = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer_y_train = imputer_y_train.fit(y_train[:])
y_train[:] = imputer_y_train.transform(y_train[:])

#importing testing data
dataset_test = pd.read_csv("credit_test.csv")
dataset_test['Months since last delinquent'].fillna(0,inplace=True)
X_test = dataset_test.iloc[:,2:17].values
y_test = dataset_test.iloc[:,17].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
label_encoder_exp = LabelEncoder()
X_test[:, 4] = label_encoder_exp.fit_transform(X_test[:, 4].astype(str))
label_encoder_term = LabelEncoder()
X_test[:, 1] = label_encoder_term.fit_transform(X_test[:, 1].astype(str))
label_encoder_purpose = LabelEncoder()
X_test[:, 6] = label_encoder_purpose.fit_transform(X_test[:, 6].astype(str))
label_encoder_home_ownership = LabelEncoder()
X_test[:, 5] = label_encoder_home_ownership.fit_transform(X_test[:, 5].astype(str))

#Handling missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, 0:1])
X_test[:, 0:1] = imputer.transform(X_test[:, 0:1])

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_test[:, 1:2])
X_test[:, 1:2] = imputer.transform(X_test[:, 1:2])

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, 2:4])
X_test[:, 2:4] = imputer.transform(X_test[:, 2:4])

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_test[:, 4:7])
X_test[:, 4:7] = imputer.transform(X_test[:, 4:7])

imputer_bankruptcy = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer_bankruptcy = imputer_bankruptcy.fit(X_test[:, 14:15])
X_test[:, 14:15] = imputer_bankruptcy.transform(X_test[:, 14:15])

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_test[:, 7:8])
X_test[:, 7:8] = imputer.transform(X_test[:, 7:8])

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_test[:, 8:9])
X_test[:, 8:9] = imputer.transform(X_test[:, 8:9])

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_test[:, 10:12])
X_test[:, 10:12] = imputer.transform(X_test[:, 10:12])

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, 12:14])
X_test[:, 12:14] = imputer.transform(X_test[:, 12:14])

y_test = y_test.reshape(-1,1)
imputer_y_test = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer_y_test = imputer_y_test.fit(y_test[:])
y_test[:] = imputer_y_test.transform(y_test[:])

#Construction of ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
classifier = Sequential()
classifier.add(Dense(activation='relu',units = 8, input_dim = 15, kernel_initializer = 'glorot_uniform' ))
classifier.add(Dropout(rate = 0.2))
classifier.add(Dense(activation='relu',units = 8, kernel_initializer = 'glorot_uniform'))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(activation='sigmoid',units = 1, kernel_initializer = 'glorot_uniform'))
classifier.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,batch_size=10, epochs=100)

#predicting the result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) 
 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
