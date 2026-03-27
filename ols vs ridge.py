import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import linear_model

# dataset load
df = sns.load_dataset("tips")

# feature & target
X = df[["total_bill"]].values
y = df["tip"].values

# ছোট sample নেওয়া (2টা point only, original concept maintain করার জন্য)
X_train = X[:2]
y_train = y[:2]

# test data
X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

np.random.seed(0)


classifiers = dict(
    ols=linear_model.LinearRegression(),
    ridge=linear_model.Ridge(alpha=1)
)

for name, clf in classifiers.items():

    fig, ax = plt.subplots(figsize=(5, 4))

    for _ in range(6):
      
        # noise add
        this_X = X_train + 0.5 * np.random.normal(size=X_train.shape)

        clf.fit(this_X, y_train)

        ax.plot(X_test, clf.predict(X_test), color="gray")
        ax.scatter(this_X, y_train, s=20, color="gray")

    # original data train
    clf.fit(X_train, y_train)

    ax.plot(X_test, clf.predict(X_test), color="blue", linewidth=2)
    ax.scatter(X_train, y_train, color="red", s=50)

    ax.set_title(name)
    ax.set_xlabel("Total Bill")
    ax.set_ylabel("Tip")

plt.show()