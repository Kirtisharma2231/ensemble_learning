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

# ===================== MODEL 1: DECISION TREE =====================

dt_model=DecisionTreeClassifier(max_depth=4,random_state=42)
dt_model.fit(X_train,y_train)
y_pred_dt=dt_model.predict(X_test)
print("======== TEST DATA ACCURACY =========")
print("\n")
print("Decison tree Accuracy : ",round(accuracy_score(y_test,y_pred_dt)*100,2),"%")

# ===================== MODEL 2: BAGGING =====================

bag=BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=500,max_samples=0.5,bootstrap=True,random_state=42)
bag.fit(X_train,y_train)
y_pred_bc=bag.predict(X_test)
print("Bagged Decison tree Accuracy : ",round(accuracy_score(y_test,y_pred_bc)*100,2),"%")

# ===================== MODEL 3: RANDOM FOREST =====================

rf=RandomForestClassifier(max_depth=5,max_features=2,n_estimators=500,random_state=42)
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)
print("Random Forest Decison tree Accuracy : ",round(accuracy_score(y_test,y_pred_rf)*100,2),"%")

# ===================== MODEL 4: SVM + BAGGING =====================

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

# ===================== CONCLUSION =====================

print("\nConclusion:")
print("Ensemble models (Bagging & Random Forest) performed better than a single Decision Tree, also making it the best-performing model for this dataset.")
print("Bagged SVM achieved the highest accuracy in testing dataset , showing better generalization on this dataset.")
print("Ensemble models outperformed the single Decision Tree, with bagged SVM giving the best accuracy and overall most reliable performance. ")


