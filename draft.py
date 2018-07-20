from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# preparing data
x_data = []
y_data = []
with open('ionosphere.data', 'r') as file:
    for line in file:
        y_data.append(line.split(',')[34].strip())
        x_data.append([float(x) for x in line.split(',')[:34]])
n_splits = [2, 4, 6, 8, 10]
for n in n_splits:
    # Fit  model
    clf1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1))
    clf2 = AdaBoostClassifier(base_estimator=KNeighborsClassifier(5))
    scores = cross_val_score(clf1, x_data, y_data, cv=n)
    scores2 = cross_val_score(clf2, x_data, y_data, cv=n)
    print('clf1 scores = ', scores.mean(), ' clf2 scores = ', scores2.mean())

