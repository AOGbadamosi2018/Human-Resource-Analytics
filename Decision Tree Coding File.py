# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:12:05 2020

@author: AOG
"""

#CLASSIFICATION ALGORIGTHM
#classificationn decison tree

# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset as well as independent and dependent variables
dataset = pd.read_csv("Iris.csv")
X = dataset.iloc[:,0:5].values
Y = dataset.iloc[:, 5].values


#Encoding Categorical Data(Dependent Variable) - since inputs to the algorithm have to be numerical



#encode the dependent variables as:
#0 -Iris-setosa,1 - Iris-versicolor,Iris-virginica-2

from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
Y=le.fit_transform(Y)


# Split dataset into train 70% and test set30%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30,random_state=0)


#building the model and fitting to our data
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)

#testing the model
y_pred=classifier.predict(X_test)

#Analysing the model performance
#build confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test,y_pred)
