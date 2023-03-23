import torch
import matplotlib.pyplot as plt

# quadratic model y^=w1x^2 + w2x + b

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
w1 = torch.tensor([1.0]) # w1的初值为1.0
w1.requires_grad = True # 需要计算梯度

w2 = torch.tensor([1.0]) # w2的初值为1.0
w2.requires_grad = True # 需要计算梯度

b = torch.tensor([1.0]) # b的初值为1.0
b.requires_grad = True # 需要计算梯度

def forward(x):
    return w1*x*x + w2*x + b # w是一个Tensor
 
 
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2
 
print("predict (before training)", 4, forward(4).item())

epoch_list = [] # 存起来好画图
cost_list = []

w1_list = []
w2_list = []
b_list = []

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l =loss(x,y) # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        l.backward() #  backward,compute grad for Tensor whose requires_grad set to True
        print('\tgrad:', x, y, w1.data, w2.data, b.data)
        w1.data = w1.data - 0.01 * w1.grad.data   # 权重更新时，注意grad也是一个tensor
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data

        w1.grad.data.zero_() # after update, remember set the grad to zero
        w2.grad.data.zero_() 
        b.grad.data.zero_() 
 
    print('progress:', epoch, l.item()) # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）
    epoch_list.append(epoch)

    cost_list.append(l.item())
    w1_list.append(w1.item())
    w2_list.append(w2.item())
    b_list.append(b.item())
 
print("predict (after training)", 4, forward(4).item())


plt.figure(1)
plt.subplot(1, 2, 1) #图一包含1行2列子图，当前画在第一行第一列图上
plt.plot(epoch_list,cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')

plt.figure(1)
plt.subplot(1, 2, 2)#当前画在第一行第2列图上
plt.plot(epoch_list,w1_list, "g", label="w1")
plt.plot(epoch_list,w2_list, "r", label="w2")
plt.plot(epoch_list,b_list, label="b")
plt.legend()
plt.xlabel("epoch")
plt.show() #两张图一起show



# plt.figure(1)
# plt.plot(epoch_list,cost_list)#画在图1上
# plt.ylabel('cost')
# plt.xlabel('epoch')
# plt.show() 

# plt.figure(2)
# plt.plot(epoch_list,w1_list, "g", label="w1")#画在图2上，且不在一个窗口
# plt.plot(epoch_list,w2_list, "r", label="w2")
# plt.plot(epoch_list,b_list, label="b")
# plt.legend()
# plt.xlabel("epoch")
# plt.show() # 关了第一张才能出现第二张

