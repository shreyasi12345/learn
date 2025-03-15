import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris=load_iris()

X=iris.data
Y=iris.target

model=SVC()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
print(f"Accuracy: {accuracy_score(Y_test, y_pred):.2f}")



