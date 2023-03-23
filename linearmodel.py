# 线性模型 y_hat = w * x 
import numpy as np
import matplotlib.pyplot as plt

def forward(x): #前馈,Define the model
    return x*w

def loss(x, y): #损失,Define the loss function
    y_pred = forward(x)
    return (y_pred - y)**2

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 穷举法
w_list = [] #List w_list save the weights 𝝎
mse_list = [] #List mse_list save the cost values of each 𝝎

for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val #Value of cost function is the sum of loss function
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)

plt.plot(w_list,mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()  