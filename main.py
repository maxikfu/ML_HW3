from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
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
error_rate = []
for n in n_splits:
    kf = KFold(n_splits=n)
    for train, test in kf.split(x_data):
        x_train = np.array(x_data)[train]
        y_train = np.array(y_data)[train]
        x_test = np.array(x_data)[test]
        y_test = np.array(y_data)[test]
        # Fit  model
        model_1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                 algorithm="SAMME",
                                 n_estimators=200)
        model_1.fit(x_train, y_train)


        # Predict
        y_1 = model_1.predict(x_test)

        correct = 0
        wrong = 0
        for i,j in zip(y_1, y_test):
            if i == j:
                correct += 1
            else:
                wrong += 1
        error_rate.append(wrong /(wrong + correct))
    print(np.mean(error_rate))
