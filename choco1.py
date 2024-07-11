import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


#here we load the data
data = pd.read_csv('flavors_of_cacao.csv')

#now lets inspect data a bit
#this prints the first five rows to give some insight
#print(data.head())
#this tells us about number of entries, column names, etc
#print(data.info())
#describe tells us about count,mean,standard deviation, min,max...
#print(data.describe())

#check for missing values
#print(data.isnull().sum())


#I have to delete all values that are not numeric for heatmap plot
numeric_data = data.select_dtypes(include=['float64', 'int64'])

#plt.figure(figsize=(12, 10))
#sns.heatmap(numeric_data.corr(), annot=True)
#plt.show()

#as we have seen, REF and review date have no correlation with rating
data.drop(['REF', 'Review\nDate'], axis=1, inplace=True)

#PREPROCESSING

data = data.dropna(axis=0)
#print(data.isnull().sum())

#print(data.dtypes)

#lets rename some columns, so its easier for me to handle data
#print(data.columns)
data.columns = ['Company', 'SpecificOrigin', 'CocoaPercent', 'Location', 'Rating' ,'BeanType', 'BroadOrigin'] 

#lets turn percents, that are currently strings, into floats

def removePercent(data):
 return data.apply(lambda x: float(x.strip('%'))/100)

data['CocoaPercent'] = removePercent(data['CocoaPercent'])

#print(data['CocoaPercent'])

categorical_features = ['Company', 'SpecificOrigin', 'Location', 'BeanType', 'BroadOrigin']


def onehot_encode(data, columns):
 for column in columns:
  dummies = pd.get_dummies(data[column])
  data = pd.concat([data, dummies], axis=1)
  data.drop(column, axis=1, inplace=True)
 return data

data = onehot_encode(data, categorical_features)


#Now lets split our data between y and X

y = data['Rating']
X = data.drop('Rating', axis=1)

print(X)


