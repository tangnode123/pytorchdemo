import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#处理多维度特征的输入
raw_data=np.loadtxt("./dataset/diabetes.csv.gz",delimiter=',',dtype=np.float32)

x_data=torch.from_numpy(raw_data[:,:-1])
y_data=torch.from_numpy(raw_data[:,[-1]])



class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1=torch.nn.Linear(8,6)#输入维度是8输出维度是6
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.sigmod=torch.nn.Sigmoid()
    def forward(self,x):
        x=self.sigmod(self.linear1(x))
        x=self.sigmod(self.linear2(x))
        x=self.sigmod(self.linear3(x))
        return x

model=Model()
criterion=torch.nn.BCELoss(reduction='mean')
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

for epoch in range(100):
    y_pred=model.forward(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())
    #Backward
    optimizer.zero_grad()
    loss.backward()
    #update
    optimizer.step()
