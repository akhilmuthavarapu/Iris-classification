# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:57:28 2024

@author: akhil
"""
import pickle
filename="saved_model.pkl"
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import load_iris
warnings.filterwarnings("ignore")
iris=load_iris()
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

m=[LinearRegression(),LogisticRegression(),KNeighborsClassifier(),DecisionTreeClassifier(),SVC(kernel="linear"),SVC(kernel="rbf")]
for i in m:
    i.fit(x_train,y_train)
    y_pred=i.predict(x_test)
    print(str(i).replace("()"," ----------> "),round(i.score(x_train,y_train)*100,3))
    try:
        with open(filename,'wb') as file:
            pickle.dump(i,file)
        print("model saved successfully")
    except Exception as e:
        print("error in saving the model: {e}")
