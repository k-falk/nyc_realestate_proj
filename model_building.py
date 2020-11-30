# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 21:52:33 2020

@author: Xkfal
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

path = 'C:/Users/Xkfal/Documents/nyc_realestate_proj/'
df = pd.read_csv(path + 'explored_data.csv')

df_model = df[['sale_price','borough', 'neighborhood', 'building_category', 'tax_class', 'total_units', 'distance', 'square_ft', 'age']]
df_model = df_model[~df_model.isin([np.nan, np.inf, -np.inf]).any(1)]


from data_input import data_in
np.array(data_in).reshape(301,1)

df_dumb = pd.get_dummies(df_model)

X = df_dumb.drop('sale_price', axis=1)
y = df_dumb.sale_price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor


def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))

from sklearn.metrics import mean_squared_error

rf1_regr = RandomForestRegressor()
rf1_regr.fit(X1_train, y1_train)
Y1_pred_rf = rf1_regr.predict(X1_test)
rmse(y1_test,Y1_pred_rf)

alpha=0.00099
lasso_regr1=Lasso(alpha=alpha,max_iter=50000)
lasso_regr1.fit(X1_train, y1_train)
Y1_pred_lasso=lasso_regr.predict(X1_test)
(rmse(y1_test,Y1_pred_lasso))




linreg = LinearRegression()
linreg.fit(X_train, y_train)
Y_pred_lin = linreg.predict(X_test)
rmse(y_test,Y_pred_lin)

alpha=0.00099
lasso_regr=Lasso(alpha=alpha,max_iter=50000)
lasso_regr.fit(X_train, y_train)
Y_pred_lasso=lasso_regr.predict(X_test)
np.exp1m(rmse(y_test,Y_pred_lasso))

rf_regr = RandomForestRegressor()
rf_regr.fit(X_train, y_train)
Y_pred_rf = rf_regr.predict(X_test)
np.expm1(rmse(y_test,Y_pred_rf))


import statsmodels.api as sm
x_sm = sm.add_constant(X)
model = sm.OLS(y, x_sm)
results = model.fit().summary()
results
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)
np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))


# lasso regression 
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))
alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
    
plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]


#Model Selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))


results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=21)
    cv_results = cross_val_score(model, X1_train, y1_train, scoring='neg_mean_absolute_error', cv=kfold)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)




#Grid Search
rf = RandomForestRegressor()
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)

gs.best_score_
gs.best_estimator_




#from sklearn.linear_model import Ridge
#ridge = Ridge(alpha=0.01, normalize=True)
#ridge.fit(X_train, y_train)
#Y_pred_ridge = ridge.predict(X_test)
#rmse(np.expm1(y_test),np.expm1(Y_pred_ridge))





#
###############
#def split_data_train_model(labels, data):
#    # 20% examples in test data
#    train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42
# 
#    # training data fit
#    regressor = RandomForestRegressor()
#    regressor.fit(x_data, y_data)
# 
#    return test, test_labels, regressor
#
#
#y_data, x_data, feature_names = load_input("regression_dataset.xlsx")
#x_test, x_test_labels, regressor = split_data_train_model(y_data, x_data)
# 
#predictions = regressor.predict(x_test)
#
## find the correlation between real answer and prediction
#correlation = round(pearsonr(predictions, x_test_labels)[0], 5)
# 
#output_filename = "rf_regression.png"
#title_name = "Random Forest Regression - Real House Price vs Predicted House Price - correlation ({})".format(correlation)
#x_axis_label = "Real House Price"
#y_axis_label = "Predicted House Price"
# 
## plot data
#simple_scatter_plot(x_test_labels, predictions, output_filename, title_name, x_axis_label, y_axis_label)
#
#
#
#
#
#
#
#
#######################
#
#
#
#rf = RandomForestRegressor()
#rf.fit(X_train, y_train)
#Y_pred_rf = rf.predict(X_test)
#rmse(np.expm1(y_test),np.expm1(Y_pred_lasso))
#X = X_train
#y = y_train
#
#plt.plot(X_test, rf.predict(X_test), color = 'blue')
#plt.show()
#
#
#GB = GradientBoostingRegressor()
#GB.fit(X_train, y_train)
#Y_pred_lin = GB.predict(X_test)
#np.expm1(rmse(y_test,Y_pred_lin))
#
#LR = Lasso()
#LR.fit(X_train, y_train)
#Y_pred_lin = LR.predict(X_test)
#rmse(y_test,Y_pred_lin)
#
#tpred_lr = LR.predict(X_test)
#tpred_rf = rf.predict(X_test)
#tpred_gb = GB.predict(X_test)
#
#from sklearn.metrics import mean_absolute_error
#mean_absolute_error((y_test),(tpred_rf))
#mean_absolute_error((y_test),(tpred_gb))
#mean_absolute_error(np.expm1(y_test),np.expm1(tpred_lr))
#
#mean_absolute_error(np.expm1(y_test) , np.expm1((tpred_gb+tpred_rf)/2))
#
#
###Grid Search
#rf = RandomForestRegressor()
#from sklearn.model_selection import GridSearchCV
#parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}
#
#gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
#gs.fit(X_train,y_train)
#
#gs.best_score_
#gs.best_estimator_
#
#
#from sklearn.metrics import mean_absolute_error
#
#pred_gs = lm_l.predict(X_test)
#mean_absolute_error(y_test, pred_gs)
#
#
#######################
#training, testing = train_test_split(df_dumb, test_size=0.2, random_state=0)
#
#df_train_s = training.loc[:,df_dumb.columns]
#X_train_s = df_train_s.drop(['sale_price'], axis=1)
#y_train_s = df_train_s.loc[:, ['sale_price']]
#
#df_test_s = testing.loc[:,df_dumb.columns]
#X_test_s = df_test_s.drop(['sale_price'], axis=1)
#y_test_s = df_test_s.loc[:, ['sale_price']]
#
#
#
#rf_reg = RandomForestRegressor()
#
#rf_reg.fit(X_train_s, y_train_s)
#
#y_pred_s_rf = rf_reg.predict(X_test_s)
#
## Compute 5-fold cross-validation scores: cv_scores
#cv_scores_rf = cross_val_score(rf_reg, X_train_s, y_train_s, cv=5)
#
#
#print("R^2: {}".format(rf_reg.score(X_test_s, y_test_s)))
#rmse = np.sqrt(mean_squared_error(y_test_s, y_pred_s_rf))
#print("Root Mean Squared Error: {}".format(rmse))
#
#print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores_rf)))
## Print the 5-fold cross-validation scores
#print(cv_scores_rf)
