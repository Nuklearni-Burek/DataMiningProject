import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score


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

plt.figure(figsize=(12, 10))
sns.heatmap(numeric_data.corr(), annot=True)
plt.show()

#as we have seen, REF and review date have no correlation with rating
data.drop(['REF', 'Review\nDate'], axis=1, inplace=True)
