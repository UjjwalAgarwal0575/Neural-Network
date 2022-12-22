STUDENT_ROLLNO = "IMT2020128" #yourRollNumberHere
STUDENT_NAME = "Ujjwal Agarwal" #yourNameHere

#@PROTECTED_1
##DO NOT MODIFY THE BELOW CODE. NO OTHER IMPORTS ALLOWED. NO OTHER FILE LOADING OR SAVING ALLOWED.
from torch import Tensor
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader
# import torchmetrics
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection 
import torch.utils.data as data 
import numpy as np 
import torch
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
submission = np.load("sample_submission.npy")
#@PROTECTED_1

# y_test = np.load("/kaggle/input/y-test/y_test.npy")

# Reference https://www.kaggle.com/code/riteshsinha/neural-networks-with-pytorch-mnist

num_rows,sz       = X_train.shape # observations , size of image

X_train = Tensor(X_train).cuda()
X_test = Tensor(X_test).cuda()
y_train = Tensor(y_train).cuda()
# y_test = Tensor(y_test).cuda()
# X_train, y_train, X_test, y_test = map(Tensor, (X_train, y_train, X_test, y_test)
y_train = y_train.type(torch.LongTensor)
y_train = y_train.cuda()


# Using this architecture i am getting accuracy of 54 in 25 minutes for 1000 iterations.

model_linear = nn.Sequential(nn.Linear(sz,2000), nn.ELU(),nn.Dropout(p=0.4), nn.Linear(2000,1000),nn.ELU(),nn.Linear(1000,500), nn.ELU(), nn.Linear(500,250),nn.ELU(), nn.Linear(250,500),nn.ELU(), nn.Linear(500,1000),nn.ELU(), nn.Linear(1000,2000),nn.ELU(),nn.Linear(2000,10),nn.Softmax(dim=1))

# Using this accuracy around 50 in 5 minutes with 100 iterations. 
# model_linear = nn.Sequential(nn.Linear(sz,320), nn.ReLU(), nn.Linear(320,320),nn.ReLU(),nn.Linear(320,10),nn.Softmax(dim=1))
model_linear.cuda()
opt = optim.SGD(model_linear.parameters(), lr=0.001)
loss_func = nn.functional.cross_entropy
iterations = 1000
batch_size = 100

print(model_linear)


for itr in range(iterations):
    for i in range((X_train.shape[0]-1)//batch_size + 1):
        start_i = i * batch_size
        end_i = start_i + batch_size
        x = X_train[start_i:end_i]
        y = y_train[start_i:end_i]
        pred = model_linear(x)
        loss = loss_func(pred, y)
        loss.backward()
        opt.step() # Updating weights.
        opt.zero_grad()

def accuracy(out, y_batch): 
    return (torch.argmax(out, dim=1)==y_batch).float().mean()

# print(accuracy(model_linear(X_test), y_test))


with torch.no_grad():
    output = model_linear(X_test)
submission = output.argmax(axis=1).to('cpu').numpy()


#@PROTECTED_2
np.save("{}__{}".format(STUDENT_ROLLNO,STUDENT_NAME),submission)
#@PROTECTED_2u