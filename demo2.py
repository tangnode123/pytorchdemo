import numpy as np
import matplotlib.pyplot as plt
# 穷举法
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
def forward(x):
    return x*w
def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)*(y_pred-y)
w_list=[]
mse_list=[]
for w in np.arange(0.0,4.1,0.1):
    print('w=',w)
    l_sum=0
    for x_val,y_val in zip(x_data,y_data):
        y_pred_val=forward(x_val)
        loss_val=loss(x_val,y_val)
        l_sum+=loss_val
        print('\t',x_val,y_val,y_pred_val,loss_val)
    print("MSE=",l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False
plt.plot(w_list,mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.title('线性模型')
plt.show()