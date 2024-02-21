import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

#数字0~9识别
batch_size=64
transform=transforms.Compose([
    transforms.ToTensor(),#单通道，把取到的图像转化为1*28*28的张量，介于0到1之间
    transforms.Normalize((0.1307, ),(0.3081, ))
])
train_dataset=datasets.MNIST(root='./dataset/mnist/',train=True,download=True,transform=transform)

train_loader=DataLoader(train_dataset,shuffle=True,batch_size=batch_size)

test_dataset=datasets.MNIST(root='./dataset/mnist/',train=False,download=True,transform=transform)

test_loader=DataLoader(test_dataset,shuffle=False,batch_size=batch_size)

class MinistModel(torch.nn.Module):
    def __init__(self):
        super(MinistModel,self).__init__()
        self.l1=torch.nn.Linear(784,512)
        self.l2=torch.nn.Linear(512,256)
        self.l3=torch.nn.Linear(256,128)
        self.l4=torch.nn.Linear(128,64)
        self.l5=torch.nn.Linear(64,10)
    def forward(self,x):
        x=x.view(-1,784)
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=F.relu(self.l4(x))
        x=self.l5(x)
        return x

model=MinistModel()
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)#momentum=0.5:这是动量(momentum)参数，用于加速SGD的收敛过程并减少振荡。动量越大，平滑效果越好，但也可能导致训练不稳定。

def MnistTrain(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        optimizer.zero_grad()

        outputs=model(inputs)
        loss=criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if batch_idx%300==299:
            print('[%d,%5d] loss:%.3f'%(epoch+1,batch_idx+1,running_loss/300))
            running_loss=0.0

def MnistTest():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            outputs=model(images)
            _,predicted=torch.max(outputs.data,dim=1)#dim沿着第一个维度，返回的是最大值和最大值下标
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print('Accuracy on test set:%d %%'%(100*correct/total))

#提取特征的思考FFT
if __name__=='__main__':
    for epoch in range(10):
        MnistTrain(epoch)
        MnistTest()