import numpy as np
import matplotlib.pyplot as plt
#随机梯度下降算法练习
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
w=1.0
def forward(x):
    return x*w
def loss(xs,ys):
    y_pred=forward(x)
    return (y_pred-y)**2
def gredient(x,y):
    return 2*x*(x*w-y)
print("Predict (before training)",4,forward(4))
cost_list=[]
epoch_list=[]
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        gred_val=gredient(x,y)
        w-=0.01*gred_val
        print("\tgrad:",x,y,gred_val)
        loss_val=loss(x,y)
        cost_list.append(loss_val)
        epoch_list.append(epoch)
    print('Epoch:',epoch,'w=',w,'loss=',loss_val)
print("Predict (after training)",4,forward(4))
plt.plot(epoch_list,cost_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
print("请输入x值")
x1=float(input())
pred=forward(x1)
print("预测的结果为",pred)