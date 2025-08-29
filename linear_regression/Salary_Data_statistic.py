import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import VarianceThreshold
import pickle

path = (r'C:\Users\WELCOME\Desktop\Python\linear_regression\Salary_Data.csv')
df = pd.read_csv(path)
df.head()

df.shape

df.columns

df.isnull().sum()

df.dtypes


X = df.drop('YearsExperience', axis=1)
y = df['Salary']

#from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1234, test_size = 0.30)

X_train.shape, X_test.shape

y_train.shape, y_test.shape

df.shape

X_train.ndim

#from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train, y_train)

y_predictions = LR.predict(X_test)

y_test.shape, y_predictions.shape


X_test.iloc[0] 
 
 
X_test.iloc[0].values

LR.predict([X_test.iloc[0].values,
 X_test.iloc[1].values])

ip1 = [5]
LR.predict([ip1])

X_test.shape, y_test.shape, y_predictions.shape

test_data = X_test
test_data['y_actual'] = y_test
test_data['y_predictions'] = y_predictions
test_data

print(y_test.values[:5])
print(y_predictions[:5])

#from sklearn.metrics import r2_score, mean_squared_error

R2 = r2_score(y_test, y_predictions)
MSE = mean_squared_error(y_test, y_predictions)
RMSE = np.sqrt(MSE)
 
print("R-sqaure:", R2)
print("MSE:", MSE)
print("RMSE:", RMSE)

s = 0
for i in range(len(y_test)):
    v1 = y_test.values[i]-y_predictions[i]
    v2 = v1**2
    s = s+v2
print(s/len(y_test))

LR.coef_
print("The coeffiecnt of Years_of_experience is:", LR.coef_)

LR.intercept_

X_train.columns


#from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=0)

vt.fit(df)

vt.variances_

vt.get_support()

vt.get_params()

vt.threshold

cols = vt.get_feature_names_out()

df[cols]

path = r'C:\Users\WELCOME\Desktop\Python\linear_regression\Salary_Data.csv'
df = pd.read_csv(path)
df.head()

from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=0)

X = df.drop('YearsExperience',axis=1) 
 
vt.fit(X)
vt.variances_
vt.get_support()
cols = vt.get_feature_names_out()
X[cols] 

from statsmodels.api import OLS

OLS(y_train, X_train).fit().summary()


pickle.dump(LR, open('YearsExperience_model.pkl', 'wb'))

model = pickle.load(open('YearsExperience_model.pkl', 'rb'))
model