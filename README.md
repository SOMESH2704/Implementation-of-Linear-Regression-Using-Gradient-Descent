# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Someshwar Kumar
RegisterNumber: 212224240157
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X=np.c_[np.ones(len(X1)), X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
X=(data.iloc[1:, :-2].values)
print()
print(X)
print()
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:, -1].values).reshape(-1,1)
print(y)
print()
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print()
print('Name: Someshwar Kumar')
print("Register No: 212224240157")
print()
print(X1_Scaled)
print()
print(Y1_Scaled)
print()
theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled =scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
*/
```

## Output:

THETA:

<img width="720" height="150" alt="484192407-4bbe9d00-026a-4639-a3ad-a6603fa5cb9e" src="https://github.com/user-attachments/assets/fbfbba3c-7a22-4707-8981-a652de535f6d" />

X:

<img width="633" height="781" alt="484192603-0aeb3445-a552-4e2a-b717-6914e306a026" src="https://github.com/user-attachments/assets/8163c47d-85ea-4edd-b945-78dc46874e2d" />

Y:

<img width="431" height="748" alt="484192686-b5529020-5590-453d-a20e-788b0aae5269" src="https://github.com/user-attachments/assets/3c971577-f1bc-4ed6-bd66-7bd5e4c27b6f" />

X1_Scaled:

<img width="735" height="785" alt="image" src="https://github.com/user-attachments/assets/ceb0a529-cd56-4981-9025-03576a3450fc" />

Y1_Scaled:

<img width="565" height="775" alt="484194039-f8451e17-7b41-4cf4-bdb5-74789ea4f168" src="https://github.com/user-attachments/assets/fc62b828-a584-4ffe-b919-67d65bb91b56" />

Predicted value:

<img width="375" height="47" alt="484194171-5e2d3ee9-7b28-408f-9e12-87eb87e01c36" src="https://github.com/user-attachments/assets/c134aaaa-a282-4889-a398-c7a3c6986577" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
