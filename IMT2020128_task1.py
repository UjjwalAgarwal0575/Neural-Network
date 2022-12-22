STUDENT_NAME = "Ujjwal Agawral" #Put your name
STUDENT_ROLLNO = "IMT2020128" #Put your roll number
CODE_COMPLETE = True
# set the above to True if you were able to complete the code
# and that you feel your model can generate a good result
# otherwise keep it as False
# Don't lie about this. This is so that we don't waste time with
# the autograder and just perform a manual check
# If the flag above is True and your code crashes, that's
# an instant deduction of 2 points on the assignment.
#
#@PROTECTED_1_BEGIN
## No code within "PROTECTED" can be modified.
## We expect this part to be VERBATIM.
## IMPORTS 
## No other library imports other than the below are allowed.
## No, not even Scipy
import numpy as np 
import pandas as pd 
import sklearn.model_selection as model_selection 
import sklearn.preprocessing as preprocessing 
import sklearn.metrics as metrics 
from tqdm import tqdm # You can make lovely progress bars using this


## FILE READING: 
## You are not permitted to read any files other than the ones given below.
X_train = pd.read_csv("train_X.csv",index_col=0).to_numpy()
y_train = pd.read_csv("train_y.csv",index_col=0).to_numpy().reshape(-1,)
X_test = pd.read_csv("test_X.csv",index_col=0).to_numpy()

submissions_df = pd.read_csv("sample_submission.csv",index_col=0)
#@PROTECTED_1_END

#This model is giving accuracy of 96% using mnist data available.
# Running in 2 to 3 minutes.


Neurons_first_hidden_layer = 80
Neurons_Second_hidden_layer = 40

def init_params():
    W1 = np.random.rand( 784 ,Neurons_first_hidden_layer) - 0.5
    b1 = np.random.rand(Neurons_first_hidden_layer, ) - 0.5
    
    W2 = np.random.rand(Neurons_first_hidden_layer,Neurons_Second_hidden_layer) - 0.5
    b2 = np.random.rand(Neurons_Second_hidden_layer, ) - 0.5
    
    W3 = np.random.rand( Neurons_Second_hidden_layer, 10) - 0.5
    b3 = np.random.rand(10, ) - 0.5
    
    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(Z, 0)

def dReLU(z):
    return z>0

def softmax(z):
    # A = np.exp(Z) / sum(np.exp(Z))
    # return A
    # Taken this function from friend to avoid runtime warnings
    z = z - np.max(z, axis = 1).reshape(z.shape[0],1)
    return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(z.shape[0],1)



def one_hot(y):
    sz = y.shape[0]
    hot_y = np.zeros((sz, 10))
    for i in range(sz):
        hot_y[i][int(y[i])] = 1 
    return hot_y


def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = X.dot(W1) + b1
    A1 = ReLU(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = ReLU(Z2)
    Z3 = A2.dot(W3) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def backward_prop(Z1, A1, Z2, A2, A3, W1, W2, W3, X, Y, batch_size):
    error=A3-Y
    dZ3=error/batch_size
    dW3 = A2.T.dot(dZ3)
    db3 = np.sum(dZ3)
    
    dZ2 = dZ3.dot(W3.T)*dReLU(Z2)
    dW2 = (dZ2.T.dot(A1)).T
    db2 =  np.sum(dZ2)
    
    dZ1 = dZ2.dot(W2.T)*dReLU(Z1)
    dW1 = (dZ1.T.dot(X)).T
    db1 = np.sum(dZ1)
    
    
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2,W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    
    W3 = W3 - alpha * dW3  
    b3 = b3 - alpha * db3 
    return W1, b1, W2, b2, W3, b3

batch_size = 200
alpha = 0.05      #learning rate




def gradient_descent(X_train, y_train, alpha, iterations):
    total_batches = X_train.shape[0]//batch_size
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        for ind in range(total_batches-1):
            start_i = ind*batch_size
            end_i = start_i + batch_size
            x = X_train[start_i:end_i]
            y = y_train[start_i:end_i]
            Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, x)
            dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, A3, W1, W2, W3, x, y, batch_size)
            W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2,W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
    return W1, b1, W2, b2, W3, b3

X_train = X_train/255
y_train = one_hot(y_train)
W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, y_train, alpha, 100)


# mnist_test = mnist_test
# y_test = one_hot(y_test)

# x = mnist_test
# y = y_test
# z1, a1, z2, a2, z3, a3 = forward_prop( W1, b1, W2, b2, W3, b3, x)

# predictions = get_predictions(a2)
# print(get_accuracy(predictions, y))
# acc2 = np.count_nonzero(np.argmax(a3,axis=1) == np.argmax(y,axis=1)) / x.shape[0]
# print("Accuracy:", 100 * acc2, "%")


X_test = X_test/255
z1, a1, z2, a2, z3, a3 = forward_prop( W1, b1, W2, b2, W3, b3, X_test)

# res = np.zeros((a3.shape[0],))

result = np.argmax(a3,axis=1)
submissions_df = pd.DataFrame(result, columns=['label'])

# res_df = pd.DataFrame(res, columns = ['label'])
# submissions_df = res_df
#@PROTECTED_2_BEGIN 
##FILE WRITING:
# You are not permitted to write to any file other than the one given below.
submissions_df.to_csv("{}__{}.csv".format(STUDENT_ROLLNO,STUDENT_NAME))
#@PROTECTED_2_END