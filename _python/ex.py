import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filepath = '/Users/navi/Desktop/Kaggle/Titanic/_python/'

dtr = pd.read_csv(filepath + 'train.csv')
dts = pd.read_csv(filepath + 'test.csv')

#functions to polish data
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)  # Filling all the NaN values with -0.5
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)  # Creating Range for Age
    group_names = ['Unknown', 'Baby', 'Child', 'Teen', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabin(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x:x[0]) #Truncating Cabin Numbers as it shouldn't be important
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0]) #Adding Lname & NamePrefix in the data table
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df

def simplify_Embarked(df):
    df.Embarked = df.Embarked.fillna('N')
    return df


def drop_features(df):
    return df.drop(['Ticket', 'Name'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabin(df)
    df = simplify_fares(df)
    df =  simplify_Embarked(df)
    df = format_name(df)
    df = drop_features(df)
    return df

dtr = transform_features(dtr)
dts = transform_features(dts)

from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix', 'Embarked']
    df_combined = pd.concat([df_train[features], df_test[features]])

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

dtr , dts = encode_features(dtr,dts)

from sklearn.model_selection import train_test_split

X_all = dtr.drop(['Survived', 'PassengerId'], axis=1)
y_all = dtr['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier.
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9],
              'max_features': ['log2', 'sqrt','auto'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data.
clf.fit(X_train, y_train)

ids = dts['PassengerId']
predictions = clf.predict(dts.drop('PassengerId', axis=1))


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic-predictions.csv', index = False)
output.head()

