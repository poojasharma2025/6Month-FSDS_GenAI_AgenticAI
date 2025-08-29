# Polynomial regression: keep increasing degree of independent variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\WELCOME\Documents\Data Science\August 2025\25th Aug-Polynomial Regression\emp_sal.csv")

x = data.iloc[:,1:2].values
y= data.iloc[:,2].values
# problem: 6 level(6 years experience) employee quit job and new employee(6.5 years experience) try to join, so company has to predict salary for new employee
# linear model - by-default degree 1
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# linear regression visualization
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x),color = 'blue')
plt.title('Linear Regression Graph')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

# suppose 6 level employee resigned and one who has 6.5 years expereince try to join but after prediction , saary limlt goes to high to they reject employee with 6.5 exp. 

# prediction
lin_model_pred = lin_reg.predict([[6.5]]) # here salary 
print(lin_model_pred)

#  now we will build non-linear model (Polynomial model)
from sklearn.preprocessing import PolynomialFeatures
#poly_reg = PolynomialFeatures()
# the graph we get from above code is little bit errorneous so we increase the degree by 3, check x-poly in VE
#every ML algo increase accuracy by doing hyper-parameter tuning . so adding degree paramtere and changing degree is called hyper-parameter tuning 
#poly_reg = PolynomialFeatures(degree = 3)
#poly_reg = PolynomialFeatures(degree = 4)  # this give less error,best accuracy
poly_reg = PolynomialFeatures(degree = 5)  # this give best fit line and accurate prediction where salary of new employee will be 174.80lpa
x_poly = poly_reg.fit_transform(x)

# in linear model, degree incearse by 1 degree , but in poly by default degree is 2 degree, so it give more accurate prediction
poly_reg.fit(x_poly,y)
lin_reg_2 = LinearRegression() 
lin_reg_2.fit(x_poly, y)

# polynomial visualization
plt.scatter(x,y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)))
plt.title('Truth or bluff(polynomial regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show() 

# Prediction
poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)

----------------------------------------------------------

# so company can offer salary to new employee : 174878.between 6-7 level in dataset
