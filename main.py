

import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold


n_estimators = 20
# A learning rate of 1. may not be optimal for both SAMME and SAMME.R
learning_rate = 1.

# preparing data
x_data = []
y_data = []
with open('ionosphere.data', 'r') as file:
    for line in file:
        y_data.append(line.split(',')[34].strip())
        x_data.append([float(x) for x in line.split(',')[:34]])
n_splits = [2, 4, 6, 8, 10]
boosting_test_error_clf_1 = []
boosting_train_error_clf_1 = []
boosting_test_error_clf_2 = []
boosting_train_error_clf_2 = []
bagging_test_error_clf_1 = []
bagging_train_error_clf_1 = []
bagging_test_error_clf_2 = []
bagging_train_error_clf_2 = []
for n in n_splits:
    kf = KFold(n_splits=n, shuffle=True, random_state=5)
    ada_real_err_clf_1 = []
    ada_real_err_train_clf_1 = []
    ada_real_err_clf_2 = []
    ada_real_err_train_clf_2 = []
    bagging_real_err_clf_1 = []
    bagging_real_err_train_clf_1 = []
    bagging_real_err_clf_2 = []
    bagging_real_err_train_clf_2 = []
    for train, test in kf.split(x_data):
        X_train = np.array(x_data)[train]
        y_train = np.array(y_data)[train]
        X_test = np.array(x_data)[test]
        y_test = np.array(y_data)[test]
        dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
        n_clf = GaussianNB()
        ada_real = AdaBoostClassifier(
            base_estimator=dt_stump,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            algorithm="SAMME.R")
        ada_real_NB = AdaBoostClassifier(
            base_estimator=n_clf,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            algorithm="SAMME.R")
        bagging_clf_1 = BaggingClassifier(base_estimator=dt_stump, n_estimators=n_estimators)
        bagging_clf_2 = BaggingClassifier(base_estimator=n_clf, n_estimators=n_estimators)

        ada_real.fit(X_train, y_train)
        bagging_clf_1.fit(X_train, y_train)
        ada_real_NB.fit(X_train, y_train)
        bagging_clf_2.fit(X_train, y_train)

        y_pred_clf_1 = ada_real.predict(X_test)
        ada_real_err_clf_1.append(zero_one_loss(y_pred_clf_1, y_test))
        y_pred_clf_1 = ada_real.predict(X_train)
        ada_real_err_train_clf_1.append(zero_one_loss(y_pred_clf_1, y_train))

        y_pred_clf_1 = bagging_clf_1.predict(X_test)
        bagging_real_err_clf_1.append(zero_one_loss(y_pred_clf_1, y_test))
        y_pred_clf_1 = bagging_clf_1.predict(X_train)
        bagging_real_err_train_clf_1.append(zero_one_loss(y_pred_clf_1, y_train))

        y_pred_clf_2 = ada_real_NB.predict(X_test)
        ada_real_err_clf_2.append(zero_one_loss(y_pred_clf_2, y_test))
        y_pred_clf_2 = ada_real_NB.predict(X_train)
        ada_real_err_train_clf_2.append(zero_one_loss(y_pred_clf_2, y_train))

        y_pred_clf_2 = bagging_clf_2.predict(X_test)
        bagging_real_err_clf_2.append(zero_one_loss(y_pred_clf_2, y_test))
        y_pred_clf_2 = bagging_clf_2.predict(X_train)
        bagging_real_err_train_clf_2.append(zero_one_loss(y_pred_clf_2, y_train))

    boosting_test_error_clf_1.append(np.mean(ada_real_err_clf_1))
    boosting_train_error_clf_1.append(np.mean(ada_real_err_train_clf_1))
    boosting_test_error_clf_2.append(np.mean(ada_real_err_clf_2))
    boosting_train_error_clf_2.append(np.mean(ada_real_err_train_clf_2))

    bagging_test_error_clf_1.append(np.mean(bagging_real_err_clf_1))
    bagging_train_error_clf_1.append(np.mean(bagging_real_err_train_clf_1))
    bagging_test_error_clf_2.append(np.mean(bagging_real_err_clf_2))
    bagging_train_error_clf_2.append(np.mean(bagging_real_err_train_clf_2))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(n_splits, boosting_test_error_clf_1,
        label='Real AdaBoost DT Test Error',
        color='orange')
ax.plot(n_splits, bagging_test_error_clf_1,
        label='Bagging DT Test Error',
        color='brown')
ax.plot(n_splits, boosting_train_error_clf_1,
        label='Real AdaBoost DT Train Error',
        color='green')
ax.plot(n_splits, bagging_train_error_clf_1,
        label='Bagging DT Train Error',
        color='grey')
ax.plot(n_splits, boosting_train_error_clf_2,
        label='Real AdaBoost NB Train Error',
        color='red')
ax.plot(n_splits, bagging_train_error_clf_2,
        label='Bagging NB Train Error',
        color='purple')

ax.plot(n_splits, boosting_test_error_clf_2,
        label='Real AdaBoost NB Test Error',
        color='blue')

ax.plot(n_splits, bagging_test_error_clf_2,
        label='Bagging NB Test Error',
        color='black')

ax.set_ylim((0.0, 1))
ax.set_xlabel('K-Folds')
ax.set_ylabel('error rate')

leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.9)
plt.show()
