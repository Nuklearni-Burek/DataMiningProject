import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

#load the data
data = pd.read_csv('flavors_of_cacao.csv')

#Now lets inspect the data a little bit
#this gives us some insight, by printing first 5 rows
#print(data.head())
#this tells us about number of entries, column names etc
#print(data.info())
#describe tells us about count, mean, standard deviation, min, max...
#print(data.describe())





#let's remove spaces in column names, so we don't get any errors in our code
data.columns = [col.strip().replace('\n', ' ').replace(' ', '_') for col in data.columns]

#clean cocoa percent column and convert it to numeric
data['Cocoa_Percent'] = data['Cocoa_Percent'].str.replace('%', '').astype(float)

#handle missing values by filling with a placeholder or dropping
data = data.fillna('Unknown')

#encode categorical variables
for col in ['Company _(Maker-if_known)', 'Specific_Bean_Origin_or_Bar_Name', 'Company_Location', 'Bean_Type', 'Broad_Bean_Origin']:
    data[col] = data[col].astype('category').cat.codes

#transforming the rating attribute into binary classification
data['Rating_Binary'] = (data['Rating'] >= 3).astype(int)

print(data['Rating_Binary'])




#split the data into features and target
X = data.drop(columns=['Rating', 'Rating_Binary'])
y = data['Rating_Binary']

#now let's split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)
dt_predictions = decision_tree_model.predict(X_test)

# Logistic Regression
logistic_regression_model = LogisticRegression(max_iter=200, random_state=42)
logistic_regression_model.fit(X_train, y_train)
lr_predictions = logistic_regression_model.predict(X_test)

# XGBoost Classifier
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

#Evaluate the models
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_f1 = f1_score(y_test, dt_predictions)

lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_f1 = f1_score(y_test, lr_predictions)

xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_f1 = f1_score(y_test, xgb_predictions)

# Create a DataFrame to display the evaluation metrics
results_df = pd.DataFrame({
    'Model': ['Decision Tree', 'Logistic Regression', 'XGBoost'],
    'Accuracy': [dt_accuracy, lr_accuracy, xgb_accuracy],
    'F1 Score': [dt_f1, lr_f1, xgb_f1]
})

# Display the evaluation results
print(results_df)
