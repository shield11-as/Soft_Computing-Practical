import numpy as np

np.random.seed(1)
t = np.arange(0, 100, 0.1)
data = np.sin(t) + 0.1 * np.random.randn(len(t))

step = 10
X, y = [], []
for i in range(len(data) - step):
    X.append(data[i:i+step])
    y.append(data[i+step])

X = np.array(X)
y = np.array(y)

W = np.random.randn(step)
predictions = X @ W / step

print("Actual next value:", y[-1])
print("Predicted next value:", predictions[-1])
