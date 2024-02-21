import torch.nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
xy = np.loadtxt('./dataset/diabetes.csv.gz',delimiter=',',dtype=np.float32)
# 数据准备
x_data=torch.from_numpy(xy[:,:-1])
y_data=torch.from_numpy(xy[:,[-1]])
# 设计模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.sigmoid=torch.nn.Sigmoid()
    def forward(self,x):
        x=self.sigmoid(self.linear1(x))
        x=self.sigmoid(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        x=self.sigmoid(self.linear4(x))
        return x
model=Model()
# 二分类交叉熵
criterion = torch.nn.BCELoss(reduction='mean')
#优化器
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
epoch_list=[]
loss_list=[]
# 循环训练
for epoch in range(1000000):
    # Forward
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    #print(epoch,loss.item())
    # loss_list.append(loss)
    # epoch_list.append(epoch)
    # Backward
    optimizer.zero_grad()
    loss.backward()
    # Update
    optimizer.step()
    #用准确率acc评估
    if epoch%100000==99999:
        y_pred_label=torch.where(y_pred>=0.5,torch.tensor([0.1]),torch.tensor([0.0]))
        acc=torch.eq(y_pred_label,y_data).sum().item()/y_data.size(0)
        print("loss=",loss.item(),"acc=",acc)
# plt.plot(epoch_list,loss_list)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.show()