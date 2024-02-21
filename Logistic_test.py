import torch.nn
import numpy as np
import matplotlib.pyplot as plt
# 数据准备
x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[0],[0],[1]])
# 设计模型
class LogisticRegfressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegfressionModel,self).__init__()
        self.linear=torch.nn.Linear(1,1)
    def forward(self,x):
        y_pred=torch.sigmoid(self.linear(x))
        return y_pred
model=LogisticRegfressionModel()
# 二分类交叉熵
criterion = torch.nn.BCELoss(reduction='sum')
#优化器
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
# 循环训练
for epoch in range(1000):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
x=np.linspace(0,10,200)
x_t=torch.Tensor(x).view((200,1))
y_t=model(x_t)
y=y_t.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()