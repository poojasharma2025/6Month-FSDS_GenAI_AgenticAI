# This is tranformer used to fill missing value.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------------------------------
dataset = pd.read_csv(r"C:\Users\WELCOME\Documents\Data Science\August 2025\13th Aug-spyder(data preprocessing pipeline)\Data.csv")
x = dataset.iloc[:, :-1].values # x is independent variable like state , age, salary

#iloc(index location) from entire dataframe(:) , remove -1(:-1)

y = dataset.iloc[:,3].values    # y is dependent variable  ,hence  we divide the data into dv and idv

#-------------------------------------------------------------------------------------------------------------------

# introduce to scikit framework sklearn=dsa+math+linear algebra+stats+algo
# impute is transformer for fill missing value
# imputer = SimpleImputer() this line fill or transform missing value to numerical data. simple imputer is library
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()    # system by default take mean strategy to fill missing value . use ctrl+i to check stategy
#imputer = SimpleImputer(strategy='median')
#imputer = SimpleImputer(strategy='most_frequent')  

imputer = imputer.fit(x[:,1:3])

x[:,1:3] = imputer.transform(x[:,1:3])

#------------------------------------------------------------------------------------------------------------------
# How to encode categorical data & create a Dummy variable
# LebleEncode is used to tranform categorical to number 
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()

labelencoder_x.fit_transform(x[:,0])  # we can skip this line
x[:,0] = labelencoder_x.fit_transform(x[:,0])
labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)

# splitting dataset into training and testing set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8, test_size=0.2, random_state=0)  
# when we random_state=0 , x_train,xtest not changes , hence accuracy will be same. else accuracy will be diffwerent all time whenever we run code

# if you remove random state then your model not behave as accurate

# feature scaling -> use for better accuracy
 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

# normalization
from sklearn.preprocessing import Normalizer
sc_x = Normalizer()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)





















