# Naive bayes: 
# bernouliNB, MultinomialNB, GausianNB
# If the dependent variable is binary , we dont use multinomial NB

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\WELCOME\Documents\Data Science\September 2025\5th Sept-knn\3rd - KNN\Social_Network_Ads.csv")

x = df.iloc[:, [2,3]].values
y = df.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20, random_state=0)
 

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler    
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)'''

# Normalizer 


'''from sklearn.preprocessing import Normalizer    
sc = Normalizer()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)'''

# MULTINOMIAL NAIVE BAYES

from sklearn.naive_bayes import MultinomialNB    # # If the dependent variable is binary , we dont use multinomial NB
classifier = MultinomialNB()
classifier.fit(x_train, y_train)

#Gausian Naive bayes

'''from sklearn.naive_bayes import GaussianNB    # gausian NB and tree algo doesnt required feature scaling like standardization and normalization
classifier = GaussianNB()
classifier.fit(x_train, y_train)'''

# Training the naive bayes model on training set

'''from sklearn.naive_bayes import BernoulliNB    
classifier = BernoulliNB()
classifier.fit(x_train, y_train)'''

y_pred = classifier.predict(x_test)

# making the confusion metrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 
print(cm)

# This is to get the Models Accuracy 
from sklearn.metrics import accuracy_score  
ac = accuracy_score(y_test, y_pred)
print(ac)

# This is to get the Classification Report

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

# training score
bias = classifier.score(x_train, y_train)
print(bias)
# testing score
variance = classifier.score(x_test, y_test)
print(variance)
