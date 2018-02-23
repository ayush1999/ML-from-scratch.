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

def train(X, Y):
    temp = np.dot(transpose(X), X)
    temp = inverse(temp)
    temp = np.dot(temp, transpose(X))
    return np.dot(temp, Y)

def multiply(a, b):
    return np.dot(a, b)

if __name__=="__main__":
    filename = input("Enter the name of the file. ")
    X = read("train.csv")
    Y= []
    for ele in X:
        Y.append(ele[-1])
    X = np.ndarray.tolist(X)
    for ele in X:
        ele.remove(ele[-1])
    res = []
    for ele in X:
        ele = [1] + ele
        res.append(ele)
    X = res
    X = np.array(X)
    res = train(X, Y)
    res1 = np.array(res)
    print("The estimated values are: {}".format(res))

    # test
    X = read("test.csv")
    Y = []
    for ele in X:
        Y.append(ele[-1])
    X = np.ndarray.tolist(X)
    for ele in X:
        ele.remove(ele[-1])
    res = []
    for ele in X:
        ele = [1] + ele
        res.append(ele)
    X = res
    X = np.array(X)

    loss=0

    for i in range(len(X)):
        loss = loss + (np.dot(transpose(res1), X[i]) - Y[i])**2

    print("Net loss is: {}".format(loss))

