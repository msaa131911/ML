#problem:
'''একটি টেলিকমিউনিকেশন কোম্পানি তাদের গ্রাহকদের ছেড়ে চলে যাওয়ার ঝুঁকি (churn) কমাতে চায়। তাদের কাছে গ্রাহকদের আচরণের ঐতিহাসিক তথ্য রয়েছে এবং তারা এমন একটি মডেল তৈরি করতে
চায় যা ভবিষ্যদ্বাণী করতে পারবে যে কোন গ্রাহকদের ছেড়ে চলে যাওয়ার সম্ভাবনা সবচেয়ে বেশি।'''


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report

data={'age':[30,40,50,60,70,80,50,90],
      'monthlycharge':[50,60,80,90,100,120,80,150],
      'churn':[1,0,1,0,1,0,1,0]
      }
df=pd.DataFrame(data)
#print(df)
x=df[['age','monthlycharge']]
y=df[['churn']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

svc_model=SVC(kernel='linear',C=1.0) #c=1.0 default regularization
svc_model.fit(x_train,y_train)

y_pred=svc_model.predict(x_test)

print("Accuracy: ",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

user_age=float(input("Enter the age: "))
user_monthly_charge=float(input("Enter the monthly charge: "))

user_input=np.array([[user_age,user_monthly_charge]])

prediction=svc_model.predict(user_input)
if prediction[0]==0:
    print('the customer will churn')
else:
    print('the customer will not churn')