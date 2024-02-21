import numpy as np
import torch
from torch.utils.data import Dataset#抽象类不能去实例化
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy=np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.len=xy.shape[0]
        self.x_data=torch.from_numpy(xy[:,:-1])
        self.y_data=torch.from_numpy(xy[:,[-1]])

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len

dataset=DiabetesDataset("./dataset/diabetes.csv.gz")
train_loader=DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=0)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1=torch.nn.Linear(8,6)
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

if __name__=='__main__':
    for epoch in range(100):
        #enumerate的作用就是对可迭代的数据进行标号并将其里面的数据和标号一并打印出来。
        for i,data in enumerate(train_loader,0):
            inputs,labels=data

            y_pred=model(inputs)
            loss=criterion(y_pred,labels)
            print(epoch,i,loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
