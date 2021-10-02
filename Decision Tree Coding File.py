# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:12:05 2020

@author: AOG
"""

#BINARY CLASSIFICATION ALGORIGTHM - this means only two outcomes,(0) or (1)
#classificationn decison tree - dimple machine learning algorithm

# Importing libraries - this enables us to build work on previous work 

import numpy as np #a.k.a numerical python brings mathematical functions to code
import pandas as pd #data processing librariy for text importation from csv im this case
import matplotlib.pyplot as plt # library for plotting graph

# Importing dataset as well as independent and dependent variables
dataset = pd.read_csv("Iris.csv") #reading from the csvfile here into a structure known as a "Dataframe"
X = dataset.iloc[:,0:5].values #Ignoring the data not relevant, for example ignoring the index column
Y = dataset.iloc[:, 5].values# here picking all rows,indicated by ":" and only column with index 5

#Next section of the code
#Encoding Categorical Data(Dependent Variable) - since inputs to the algorithm have to be numerical



#encode the dependent variables as:
#0 -Iris-setosa,1 - Iris-versicolor,Iris-virginica-2

from sklearn.preprocessing import LabelEncoder #sklearn - scientific learning kit contains ready-made machine learning algorithms see line 41
le =LabelEncoder() #this creates an empty container label outputs to either of our cases i.e cat becomes 1 and dog becomes 2
Y=le.fit_transform(Y) #here we filled the expected output from line 20


# Split dataset into train 70% and test set30%
from sklearn.model_selection import train_test_split #thi helps us to divide the data 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30,random_state=0) #we specify 30% data for testing and 70% for training 


#building the model and fitting to our data
from sklearn.tree import DecisionTreeClassifier #please see line 30
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0) #we create and empty model and specify we eant to use the entropy method
classifier.fit(X_train,Y_train) #this puts the split data into the algorithm

#testing the model
y_pred=classifier.predict(X_test) #we can test our work here 

#Analysing the model performance
#build confusion matrix
from sklearn.metrics import confusion_matrix #this rectangle shows the number of right and wrong predictions
cm= confusion_matrix(Y_test,y_pred) #it compares the predicted values with the true value
