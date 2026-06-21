from sklearn.neighbors import KDTree
import numpy as np

# Dataset (x, y points)
X = np.array([
    [2, 3],
    [5, 4],
    [9, 6],
    [4, 7],
    [8, 1],
    [7, 2]
])

# KDTree build
tree = KDTree(X)

# Query point
query = np.array([[6, 3]])

# nearest 2 points
dist, ind = tree.query(query, k=2)

print("Nearest Index:", ind)
print("Distance:", dist)
print("Nearest Points:", X[ind[0]])