import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
dataset = pd.read_csv(r"C:\Users\WELCOME\Documents\Data Science\August 2025\14th Aug-linear_spyder\Salary_Data.csv")

# Independent & dependent variables
x = dataset.iloc[:, :-1]   # Features
y = dataset.iloc[:, -1]    # Target

# Fill missing numerical values
imputer = SimpleImputer()
x.iloc[:, :] = imputer.fit_transform(x)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict
y_pred = regressor.predict(x_test)

# Plot
plt.scatter(x_test.iloc[:, 0], y_test, color='red')
plt.plot(x_train.iloc[:, 0], regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Coefficients
m = regressor.coef_
c = regressor.intercept_
yr_12 = m[0] * 12 + c
print(f"Predicted salary for 12 years experience: {yr_12}")


bias = regressor.score(x_train,y_train)
bias
variance = regressor.score(x_test,y_test)
variance
# Mean
dataset.mean()                       #   this will give mean of entire dataframe 
dataset['Salary'].mean()             # this will give us mean of that particular column

#median
dataset.median()                     # this will give median of entire dataframe  
dataset['Salary'].median()           # this will give us median of that particular column
 
# mode
dataset['Salary'].mode()

#  variance
dataset.var()                         # this will give variance of entire dataframe 
dataset['Salary'].var()                # this will give us variance of that particular column        
 
# Standard deviation
dataset.std()                             # this will give standard deviation of entire dataframe
dataset['Salary'].std()                   # this will give us standard deviation of that particular column


# Coefficient of variation(cv)
# for calculating cv we have to import a library firs
from scipy.stats import variation
variation(dataset.values)                   # this will give cv of entire dataframe 
variation(dataset['Salary'])                    # this will give us cv of that particular column 

# correlation

dataset.corr()                 # this will give correlation of entire dataframe

dataset['Salary'].corr(dataset['YearsExperience'])  # this will give us correlation between these two v


#skewness

dataset.skew()                  # this will give skewness of entire dataframe  
dataset['Salary'].skew()          # this will give us skewness of that particular column

# standard error

dataset.sem()                        # this will give standard error of entire dataframe   
dataset['Salary'].sem()                # this will give us standard error of that particular colum




# Z -Score

 # for calculating Z-score we have to import a library first

import scipy.stats as stats
dataset.apply(stats.zscore)               # this will give Z-score of entire dataframe 

stats.zscore(dataset['Salary'])          # this will give us Z-score of that particular column

# degrree of freedom

a=dataset.shape[0]                # this will gives us no.of rows
b=dataset.shape[1]                # this will give us no.of columns

degree_of_freedom =a-b
print(degree_of_freedom)

# Sum of Squares Regression (SSR)

y_mean = np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)

#  Sum of Squares Error (SSE)
y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

# Sum of Squares Total (SST)
mean_total = np.mean(dataset.values)
SST= np.sum((dataset.values-mean_total)**2)
print(SST)

#  R-Square
r_square=1-SSR/SST
print(r_square)
# please refer stats code sheet with eg pdf file dated 18th aug for more info.