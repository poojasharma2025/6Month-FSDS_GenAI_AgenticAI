# Decision Tree- doesnt required feature scaling
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
 

# Feature Scaling  -- Tree algo doesnt require scaling
'''from sklearn.preprocessing import StandardScaler    
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)'''  # with scaling 91.25 and without 93.75

# Normalizer 


'''from sklearn.preprocessing import Normalizer    
sc = Normalizer()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)'''

# Training the Decision Tree classification model on the Training set- rebuild model

from sklearn.tree import DecisionTreeClassifier   
classifier = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=None)
classifier.fit(x_train, y_train)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=4, n_estimators=100)
classifier.fit(x_train, y_train)

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
