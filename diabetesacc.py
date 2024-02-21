import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

raw_data=np.loadtxt('./dataset/diabetes.csv.gz',delimiter=',',dtype=np.float32)
print(raw_data.shape)
x=raw_data[:,:-1]
y=raw_data[:,[-1]]
Xtrain,Xtest,Ytrain,Ytest=train_test_split(x,y,test_size=0.3)
Xtest=torch.from_numpy(Xtest)
Ytest=torch.from_numpy(Ytest)
class DiabetesDataset(Dataset):
    def __init__(self,data,label):
        self.len=data.shape[1]
        self.x_data=torch.from_numpy(data)
        self.y_data=torch.from_numpy(label)
    def __getitem__(self, item):
        return self.x_data[item],self.y_data[item]
    def __len__(self):
        return self.len
train_dataset=DiabetesDataset(Xtrain,Ytrain)
print(Xtrain.shape)
train_loader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,num_workers=0)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.Sigmoid()
    def forward(self,x):
        x=self.sigmoid(self.linear1(x))
        x=self.sigmoid(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        return x
model=Model()

criterion=torch.nn.BCELoss(reduction='mean')
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

def train(epoch):
    train_loss=0.0
    count=0
    for i,data in enumerate(train_loader,0):
        inputs,labels=data
        y_pred=model(inputs)
        loss=criterion(y_pred,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        count=i
    if epoch%2000 == 1999:
        print("train loss:",train_loss/count,end='')
def atest():
    with torch.no_grad():
        y_pred=model(Xtest)
        y_pred_label=torch.where(y_pred>=0.5,torch.tensor([1.0]),torch.tensor([0.0]))
        acc=torch.eq(y_pred_label,Ytest).sum().item()/Ytest.size(0)
        print("test acc:",acc)
if __name__=="__main__":
    for epoch in range(50000):
        train(epoch)
        if epoch%2000==1999:
            atest()