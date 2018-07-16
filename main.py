from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt


bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
