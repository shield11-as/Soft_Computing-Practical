import numpy as np

A = np.array([
    [ 1, -1,  1, -1],   
    [-1,  1, -1,  1],   
    [ 1,  1, -1, -1]    
])

B = np.array([
    [ 1, -1, -1],   
    [-1,  1, -1],   
    [-1, -1,  1]   
])

W = np.zeros((A.shape[1], B.shape[1]))

for i in range(len(A)):
    W += np.outer(A[i], B[i])

print("Weight Matrix W:\n", W)

A_noisy = np.array([1, -1, -1, -1])

B_recalled = np.sign(A_noisy @ W)

B_recalled[B_recalled == 0] = 1

print("\nNoisy Input A:", A_noisy)
print("Recalled Output B:", B_recalled)


A_recalled = np.sign(B_recalled @ W.T)
A_recalled[A_recalled == 0] = 1

print("Recalled Input A:", A_recalled)
