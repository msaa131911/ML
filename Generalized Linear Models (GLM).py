#Generalized Linear Models (GLM)
# GLM (Logistic Regression) 
# Binary classification
# Probability-based decision
# Real dataset simulation
# sklearn model pipeline
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
np.random.seed(42)

data_size = 200

study_hours = np.random.randint(0, 10, data_size)
attendance = np.random.randint(50, 100, data_size)
previous_marks = np.random.randint(30, 100, data_size)


#  more study + attendance + marks => pass
Y = ((study_hours*5 + attendance*0.3 + previous_marks*0.5) > 80).astype(int)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_marks": previous_marks,
    "result": Y
})

#print(df.head())
X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
simple=np.array([[5, 80, 70]]) # 5 hours study, 80% attendance, 70 marks
result=model.predict(simple)
#probability
if result[0]==1:
    print("Pass")
else:
    print("Fail")