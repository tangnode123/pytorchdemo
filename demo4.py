import torch
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
#设置权重为1.0
w=torch.tensor([1.0])
w.requires_grad=True#设置w是需要计算梯度的

def forward(x):
    return x*w

def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2

print("prdict (Before training)",4,forward(4).item())

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l=loss(x,y)
        l.backward()#每进行一次方向传播就会释放计算图
        print('\tgrad:',x,y,w.grad.item())#.item()用于在只包含一个元素的tensor中提取值，注意是只包含一个元素，否则的话使用.tolist()
        #grad也是一个tensor，要取出tensor中的data
        w.data=w.data-0.01*w.grad.data

        w.grad.data.zero_()#释放之前计算的梯度
    print("progress:",epoch,l.item())

print("predict (after training)",4,forward(4).item())