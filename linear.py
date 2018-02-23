import numpy as np

"""
For a linear regression problem with n features, 
we assume that the solution is of the form:

    h(x) = transpose(Θ)*x

where, Θ are the values we have to calculate and 
x is the given feature vector.

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

def theta_final(X, Y):
    return inverse(transpose(X) * X) * transpose(X) * Y

def multiply(a, b):
    if (a.shape[1] == b.shape[0]):
        temp = np.zeros((a.shape[0], b.shape[1]))
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                sum = 0
                for k in range(a.shape[1]):
                    sum = sum + a[i][k]*b[k][j]
                temp[i][j] = sum
        return temp
    else:
        raise(ValueError("Incorrect Dimensions"))

if __name__=="__main__":
    filename = input("Enter the name of the file. ")
    X = read("dataset.csv")
    Y= []
    for ele in X:
        Y.append(ele[-1])
    print(Y)
    X = np.ndarray.tolist(X)
    for ele in X:
        ele.remove(ele[-1])
    res = []
    for ele in X:
        ele = [1] + ele
        res.append(ele)
    X = res
    X = np.array(X)
    print("Desired values of theta vector are:{}".format(theta_final(X, Y)))
    
