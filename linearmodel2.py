# 线性模型 y_hat = w * x + b
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def forward(x): # 前馈
    return x * w + b

def loss(x, y): #损失
    y_pred = forward(x)
    return (y_pred - y)**2

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 穷举法
B = np.arange(-2.0, 2.1, 0.1)
W = np.arange(0.0, 4.1, 0.1)
w,b = np.meshgrid(W, B)#用于三维曲面的分格线座标；产生“格点”矩阵
#此处直接使用矩阵进行计算

l_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val)
    loss_val = loss(x_val, y_val)
    l_sum += loss_val
    print(y_pred_val)
    print(loss_val)
    
print('MSE=', l_sum/3)

# print("网格化后的w=",w)
# print("X维度信息",w.shape)
# print("网格化后的b=",b)
# print("Y维度信息", b.shape)

fig = plt.figure()
ax = Axes3D(fig)
plt.xlabel('w')
plt.ylabel('b')
surf = ax.plot_surface(w, b, l_sum/3, cmap=plt.get_cmap('rainbow'))  # 设置颜色映射
# 设置Z轴范围
ax.set_zlim(0, 40)
# 设置标题
plt.title("Cost Value")
plt.show()