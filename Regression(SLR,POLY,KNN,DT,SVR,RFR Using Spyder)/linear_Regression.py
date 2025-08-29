# IMPORT LIBRARY
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IMPORT THE DATASET

dataset= pd.read_csv(r"C:\Users\WELCOME\Documents\Data Science\August 2025\14th Aug-linear\Salary_Data.csv")

#  INDEPENDENT VARIABLE
x=dataset.iloc[:,:-1]

# DEPENDENT VARIABLE
y=dataset.iloc[:,-1]


# # splitting dataset into training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)

# linear regression

from sklearn.linear_model import LinearRegression

# Train model
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# Predict
y_pred = regressor.predict(x_test)


# Plot
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience (test set')
plt.xlabel('years of expereince')
plt.ylabel('salary')
plt.show()


# Coefficients
m=regressor.coef_
c=regressor.intercept_

yr_12 = (m*12)+c
