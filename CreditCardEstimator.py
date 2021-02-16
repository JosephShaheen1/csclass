# 2/1/21
# Credit card churn assessor
# Credit card churn assessor detects how likely an existing customer is to become an attrited customer

import pandas as pd
import numpy as np

import sklearn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt




data = pd.read_csv("BankChurners.csv")


for i in data.columns:
    if data[i].dtype == "object":
        d = {}
        index = 0
        for j in data[i]:
            if j not in d:
                d[j] = index
                index += 1
        data[i] = data[i].apply(lambda x: d[x])

xcolumns = ['Attrition_Flag', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 'Income_Category',
            'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon']

x = data[xcolumns]
y = data["Customer_Age"]

x_train, x_test, y_train, y_test = train_test_split(x, y)

models = [GaussianNB(), BernoulliNB(), MultinomialNB(), MLPRegressor(),
          DecisionTreeRegressor(), AdaBoostRegressor(), SVR(),RandomForestRegressor(n_estimators=10)]

bars = []

for model in models:
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    bars.append((type(model).__name__, np.abs(prediction-y_test).mean()))
    #print(type(model).__name__, np.abs(prediction-y_test).mean())
    #print(type(model), (prediction == y_test).sum()/len(y_test))
    """print(confusion_matrix(y_test, prediction))"""
    #print(prediction, y_test)

plt.gcf().subplots_adjust(bottom = 0.45)
plt.xticks(rotation = 'vertical')
plt.bar([i[0]for i in bars], [i[1]for i in bars])
plt.show()

"""plt.plot([i[0]for i in bars])
plt.yticks([i[1]for i in bars], rotation = 'vertical')
plt.show()"""

"""
estimators = [100, 500, 1000]
maxfeatures = ["auto", "sqrt", "log2"]
maxdepth = [10, 20]
minsamplessplit = [2, 5]
minsamplesleaf = [1,2]

for estimator in estimators:
    for maxfeature in maxfeatures:
        for i in maxdepth:
            for j in minsamplessplit:
                for k in minsamplesleaf:
                    model = RandomForestClassifier(n_estimators=estimator, max_features=maxfeature, max_depth= i, min_samples_split= j, min_samples_leaf=k)
                    model.fit(x_train, y_train)
                    prediction = model.predict(x_test)
                    print(estimator, maxfeature, i, j ,k )
                    print(type(model), (prediction == y_test).sum() / len(y_test))

"""


