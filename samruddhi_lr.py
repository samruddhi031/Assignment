import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


BostonTrain = pd.read_csv('boston_train.csv')


BostonTrain.head()

BostonTrain.info()
BostonTrain.describe()

#ID columns is not relevant for our analysis.

BostonTrain.drop('ID', axis = 1, inplace=True)

BostonTrain.plot.scatter('rm', 'medv')

"""In the above plot, its clear to see a linear pattern. '''

# Now lets take a look how all variables relate to each other.

plt.subplots(figsize=(12,8))
sns.heatmap(BostonTrain.corr(), cmap = 'RdGy')

"""At this heatmap plot, we can do our analysis better than the pairplot.

Lets focus at the last row, where y = medv:

When shades of Red/Orange:

1) the *more red* the color is on X axis, smaller the medv. ==> *Negative correlation*

2) When *light colors: those variables at axis x and y, they dont have any relation. ==> **Zero correlation*

3) When shades of *Gray/Black* : the more black the color is on X axis, more higher the value med is. ==> *Positive correlation*
"""

# Lets plot the paiplot, for all different correlations
# Negative Correlation: When x is high y is low and vice versa., To the right less negative correlation.

sns.pairplot(BostonTrain, vars = ['lstat', 'ptratio', 'indus', 'tax', 'crim', 'nox', 'rad', 'age', 'medv'])

sns.pairplot(BostonTrain, vars = ['rm', 'zn', 'black', 'dis', 'chas', 'medv'])

'''Trainning Linear Regression Model
Define X and Y

X: Varibles named as predictors, independent variables, features.
Y: Variable named as response or dependent variable'''

X = BostonTrain[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'black', 'lstat']]
y = BostonTrain['medv']



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

lm = LinearRegression()
lm.fit(X_train,y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

'''Considering the RMSE: we can conclude that this model average error is RMSE at medv, which means RMSE *1000 in money'''

sns.displot((y_test-predictions),bins=50);

#As more normal distribution, better it is.
