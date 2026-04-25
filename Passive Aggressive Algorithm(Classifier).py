#Passive Aggressive Algorithm(Classifier)
"""Classification (Spam detection, sentiment analysis)
Regression: Real-time systems (যেমন: live data stream)"""
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# sample data
texts = [
    "Congratulations! You won $1000",
    "Hey, how are you?",
    "Limited offer buy now",
    "Let's go to class",
]

labels = [1, 0, 1, 0]  # 1=spam, 0=ham

conv_tex=TfidfVectorizer()
X=conv_tex.fit_transform(texts)
y=labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pa_clf = PassiveAggressiveClassifier(max_iter=50, C=1.0)
pa_clf.fit(X_train, y_train)

y_pred = pa_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

if y_pred[0]==1:
    print("Spam")
else:
    print("Ham")
