###### Bagging v/s Random Forest #######

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

X,y=make_classification(n_samples=10000,n_features=6,n_redundant=0,n_informative=6,n_clusters_per_class=1,random_state=42)
df=pd.DataFrame(X,columns=['V1','V2','V3','V4','V5','V6'])
df['target']=y
print(df.info())
print(df.head(5))
print(df.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

###### model 1 #######

dt_model=DecisionTreeClassifier(max_depth=4,random_state=42)
dt_model.fit(X_train,y_train)
y_pred_dt=dt_model.predict(X_test)
print("======== TEST DATA ACCURACY =========")
print("\n")
print("Decison tree Accuracy : ",round(accuracy_score(y_test,y_pred_dt)*100,2),"%")

###### model 2 #######

bag=BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=500,max_samples=0.5,bootstrap=True,random_state=42)
bag.fit(X_train,y_train)
y_pred_bc=bag.predict(X_test)
print("Bagged Decison tree Accuracy : ",round(accuracy_score(y_test,y_pred_bc)*100,2),"%")

#### model 3 #######

rf=RandomForestClassifier(max_depth=5,max_features=2,n_estimators=500,random_state=42)
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)
print("Random Forest Decison tree Accuracy : ",round(accuracy_score(y_test,y_pred_rf)*100,2),"%")

##### model 4 ######

svm_bag=BaggingClassifier(estimator=SVC(),n_estimators=500,max_samples=0.25,bootstrap=True,random_state=42)
svm_bag.fit(X_train,y_train)
y_pred_svm=svm_bag.predict(X_test)
print("SVM Bagged Decison tree Accuracy : ",round(accuracy_score(y_test,y_pred_svm)*100,2),"%")
print("\n")
print("========== TRAINING DATA ACCURACY ==========")
print("\n")
print("DT train accuracy:", dt_model.score(X_train, y_train))
print("Bagging train accuracy:", bag.score(X_train, y_train))
print("Random Forest train accuracy:", rf.score(X_train, y_train))
print("SVM Bagged train accuracy:", svm_bag.score(X_train, y_train))




# Real difference (important)
# In Bagging, max_features (if used) applies to the whole tree
# In Random Forest, max_features is applied at every split
# This is why Random Forest shows more mixed features inside one tree
# Bagging → selects features once per tree
# Random Forest → selects features again and again at each split



