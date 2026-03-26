import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("Data_Science/data/patient_churn_dataset.csv")
print(df.info())
print(df["Insurance_Type"].isnull().sum()/ len(df)*100)

#### handling the missing values #####
print(df.Insurance_Type.mode())
df["Insurance_Type"]=df["Insurance_Type"].fillna(df.Insurance_Type.mode()[0])
print(df.Insurance_Type.value_counts())
print(df.info())

##### divide the data into features adn target #####
X=df.drop(columns=['Patient_ID','Churn'],axis=1)
Y=df['Churn']
print(df.Churn.values)
print(df.Churn.value_counts()/ len(df)*100)

##### feature encoding #######
print(X.columns)
X1=pd.get_dummies(X,columns=['Gender', 'Chronic_Disease',
       'Insurance_Type'],drop_first=True,dtype=int)
print(X1.head(5))

##### train and test the data #####
from sklearn.model_selection import train_test_split
X1_train,X1_test,Y_train,Y_test=train_test_split(X1,Y,test_size=0.2)
print(len(X1_train))
print(len(X1_test))

#### Feature scaling #####
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X1_train_sc=sc.fit_transform(X1_train)
X1_test_sc=sc.transform(X1_test)
print(X1.head(3))
# print("The x_train data is : \n ",X_train_sc)
# print("The x_test data is : \n ",X_test_sc)

#### calling the kNN algorithm #######
#### MODEL 1 #####
from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier()
knn_model.fit(X1_train_sc,Y_train)
y_pred=knn_model.predict(X1_test_sc)
print(Y_test)
print("\n")
print(y_pred)
print("\n")
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))
print("Classification report: \n",classification_report(Y_test,y_pred))
print(((Y_test == y_pred).sum()))
print("\n")
##### MODEL 2 #####
from sklearn.tree import DecisionTreeClassifier
model_dtree=DecisionTreeClassifier()
model_dtree.fit(X1_train_sc,Y_train)
y_pred_dtree=model_dtree.predict(X1_test_sc)
print(accuracy_score(Y_test, y_pred_dtree))
print(confusion_matrix(Y_test, y_pred))
print("Classification report: \n",classification_report(Y_test,y_pred))
print(((Y_test == y_pred_dtree).sum()))

##### visualizing the decision tree ######
from sklearn.tree import plot_tree
plt.figure(figsize=(10,8))
plot_tree(model_dtree,filled=True,feature_names=X1_train.columns,class_names=True,max_depth=4,label='all',impurity=True)
plt.show()
