# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 13:58:27 2021

@author: AOG
"""

#importing dependencies
import pandas as pd
from pandas import DataFrame


#IMPORTATIOIN
#reading the workbook
xlsx=pd.ExcelFile('Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx')

#creating an index based on unique sheet data
k=xlsx.sheet_names
#sheet number index iterator
n=0
#loop iterator
i=1
Sheet_container=['sheet2','sheet 3']

#appending to each dataframe
while (i<3) :
    Sheet_container[n]=DataFrame()
    Sheet_container[n]=xlsx.parse(k[i])
    print(k[i]+'parsed')
    i+=1
    n+=1
    
#referencing the data in sheet 2

Sheet_container[1]


#EXPORTATION
#export names
y=('Exisiting_Employees.csv','Employees_have_left.csv')


#exporting sheet values to.csv
#iterator assignment
p=0

while (p<2) :
    Sheet_container[p].to_csv(y[p],sep=',')
    p+=1
      
print('done')

# adding the keys and the status 

Sheet_container[0]['status'] = 'hired'
Sheet_container[1]['status'] = 'left'


#encoding dept for left employees 
d= ['accounting','hr','IT' ,'management','marketing','product_mng','RandD' ,'sales','support','technical']

m = 0 
n=1
for v in range (0,10):
    Sheet_container[1]=Sheet_container[1].replace([d[m]], n)
    Sheet_container[0]=Sheet_container[0].replace([d[m]], n)
    m+=1
    n+=1

# encoding salaries
s = ['low','medium','high']
b = 0 
a=0
for t in range (0,3):
    Sheet_container[1]=Sheet_container[1].replace([s[b]], a)
    Sheet_container[0]=Sheet_container[0].replace([s[b]], a)
    b+=1
    a+=1


# joining the two sheets as 1
master_df =pd.DataFrame()

master_df = pd.concat([ Sheet_container[0],Sheet_container[1]])


master_features = list(master_df.columns.values)

master_features


#CLASSIFICATION ALGORIGTHM
# PREDICTIVE ANALYTICS USING LOGISTIC REGRESSION 

# Importing libraries


X= master_df.iloc[:,1:-1].values #excluding customer id
Y=master_df.iloc[:,-1].values





# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,shuffle=True)


#classificationn decison tree
#no feature scaling
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)


y_pred=classifier.predict(X_test)


#build confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test,y_pred)


# EXAMPLE
# predicting for a customer slice of the main data
X_test_new =master_df.iloc[11000:,1:-1].values


# predicting for the slice
y_pred_new=classifier.predict(X_test_new)

print('Done running, the model is ready for prediction, please use line 130 to select the customers for your prediction')







