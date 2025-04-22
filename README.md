# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SUNIL KUMAR P.B.
RegisterNumber:  212223040213
*/
```
```py
import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()

data.info()
data.isnull().sum()
data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

### DATA HEAD:
![image](https://github.com/user-attachments/assets/4d0ce202-54e1-4425-9060-d8e28bdb9ab4)

<br>
<br>
<br>
<br>
<br>
<br>

### DATASET INFO:
![image](https://github.com/user-attachments/assets/02c83b0d-a985-4935-9968-2e38446e15e3)

### NULL DATASET:
![image](https://github.com/user-attachments/assets/68a5cdb1-dd06-49ab-a4fb-1a4039fbbca7)

### VALUES COUNT IN LEFT COLUMN:
![image](https://github.com/user-attachments/assets/97b70af0-987c-43b5-88c8-4e22606ead28)

### DATASET TRANSFORMED HEAD:
![image](https://github.com/user-attachments/assets/f06f322a-f0fb-4157-ab08-a7818ce44b94)

### X.HEAD:
![image](https://github.com/user-attachments/assets/3dbd5544-2bf6-4f31-9b2d-96d112f5e996)

### ACCURACY:
![image](https://github.com/user-attachments/assets/22eb711f-be4e-48df-bfa7-ecac24636772)

### DATA PREDICTION:
![image](https://github.com/user-attachments/assets/34072bc8-df10-4ca6-8954-db9d016ac706)

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
