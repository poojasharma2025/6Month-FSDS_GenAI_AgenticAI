import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\WELCOME\Documents\Data Science\August 2025\25th Aug-Polynomial Regression\emp_sal.csv")

x = data.iloc[:,1:2].values
y= data.iloc[:,2].values
#non-linear-svr,knn,poly etc.
from sklearn.svm import SVR
svr_regressor = SVR()
#svr_regressor = SVR(kernel='sigmoid', degree= 5 , gamma='auto')
#svr_regressor = SVR(kernel='sigmoid', degree= 4 , gamma='scale')
#svr_regressor = SVR(kernel='poly', degree= 5 , gamma='auto')
svr_regressor = SVR(kernel='poly', degree= 5 , gamma='scale')
#svr_regressor = SVR(kernel='poly', degree= 5 , gamma='auto', C=10.0)
svr_regressor.fit(x,y)

svr_model_pred = svr_regressor.predict([[6.5]])
print(svr_model_pred)

# knn model
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model = KNeighborsRegressor()
#knn_reg_model = KNeighborsRegressor(n_neighbors=7,weights='distance')
#knn_reg_model = KNeighborsRegressor(n_neighbors=6,weights='distance')
#knn_reg_model = KNeighborsRegressor(n_neighbors=4,weights='distance')
#knn_reg_model = KNeighborsRegressor(n_neighbors=7,weights='distance')
knn_reg_model.fit(x,y)

knn_reg_pred = knn_reg_model.predict([[6.5]])
print(knn_reg_pred)

# Decision tree
from sklearn.tree import DecisionTreeRegressor
dt_reg_model = DecisionTreeRegressor(criterion='absolute_error',max_depth=10,splitter='random' )
dt_reg_model.fit(x,y)

dt_reg_pred = dt_reg_model.predict([[6.5]])
print(dt_reg_pred)

# Random forest

from sklearn.ensemble import RandomForestRegressor # ctrl+i for options
rf_reg_model = RandomForestRegressor( n_estimators=6 ,random_state=0) # get fix accuracy ,use random_state=0, otherwise accuracy will get changing after every run
rf_reg_model.fit(x,y)

rf_reg_pred = rf_reg_model.predict([[6.5]])
print(rf_reg_pred)
