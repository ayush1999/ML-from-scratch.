import numpy as np

"""
For a linear regression problem with n features, 
we assume that the solution is of the form:

    h(x) = transpose(Θ)*x

where, Θ are the values we have to calculate and 
x is the given featue vector.

Using linear algebra, we can prove that Θ can
be found out to be:

    Θ = inverse(transpose(X)*X)*transpose(X)*Y,

where Y is the target vector.
"""

def read(filename):
    f = np.genfromtxt(filename, delimiter=',')
    return f

def transpose(X):
    temp  = np.zeros((X.shape[1], X.shape[0]))
    for i in range(X.shape[1]):
        for j in range(X.shape[0]):
            temp[i][j] = X[j][i]
    return temp

def inverse(X):
    return np.linalg.inv(X)

if __name__=="__main__":
    