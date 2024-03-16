import numpy as np

x = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
y = np.array([[0, 0], [2, 2], [3, 3], [4, 4], [5, 5]]).T

print(np.dot(x, y)) # Expected: [[2 2] [8 8] [18 18] [32 32] [50 50]]

print(sum(np.dot(x, y))) # Expected: [[2 2] [8 8] [18 18] [32 32] [50 50]]