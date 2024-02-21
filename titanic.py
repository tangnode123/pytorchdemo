import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.environ["KMP_DUPLIATE_LIB_OK"]="TRUE"

trainData=pd.read_csv('./dataset/Titanic/train.csv')
label=trainData['Survived']
inputs =trainData[['Pclass','SibSp','Parch','Fare']]
y=torch.from_numpy(label.values)
x=torch.from_numpy(inputs.values)
x=x.float()
y=y.float()
class Titanic_num(torch.nn.Module):
    def __init__(self):
        super(Titanic_num,self).__init__()
        self.activate=torch.nn.ELU()
        self.sigmoid=torch.nn.Sigmoid()
        self.linear1=torch.nn.Linear(4,2)
        self.linear2=torch.nn.Linear(2,1)
    def forward(self,x):
        y_pred=self.activate(self.linear1(x))
        y_pred=self.sigmoid(self.linear2(y_pred))
        y_pred=y_pred.squeeze(-1)
        return  y_pred
model=Titanic_num()
criterion=torch.nn.BCELoss(reduction='sum')
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

epoch_list=[]
l_list=[]
for epoch in range(200):
    y_pred=model(x)
    loss=criterion(y_pred,y)
    epoch_list.append(epoch)
    l_list.append(loss.data.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
plt.plot(np.array(epoch_list),np.array(l_list))