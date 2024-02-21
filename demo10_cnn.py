import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)#通道数输入是1输出是10
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5)
        self.conv3=torch.nn.Conv2d(20,30,kernel_size=3)
        self.pooling=torch.nn.MaxPool2d(2)
        self.fc1=torch.nn.Linear(30,20)#输入特征是320输出特征是10
        self.fc2=torch.nn.Linear(20,10)
    def forward(self,x):
        batch_size=x.size(0)
        x=self.pooling(F.relu(self.conv1(x)))
        x=self.pooling(F.relu(self.conv2(x)))
        x=self.pooling(F.relu(self.conv3(x)))
        x=x.view(batch_size,-1)#flatten
        x=self.fc1(x)
        x=self.fc2(x)
        return x

model=MinistModel()

device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
model.to(device)

criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def MnistTrain(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        inputs,target=inputs.to(device),target.to(device)
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
            images, labels = images.to(device), labels.to(device)
            outputs=model(images)
            _,predicted=torch.max(outputs.data,dim=1)#dim沿着第一个维度，返回的是最大值和最大值下标
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print('Accuracy on test set:%d %%'%(100*correct/total))
    accuracy_list.append(100*correct/total)

epoch_list=[]
accuracy_list=[]
#提取特征的思考FFT
if __name__=='__main__':
    for epoch in range(10):
        MnistTrain(epoch)
        epoch_list.append(epoch)
        MnistTest()
    plt.plot(epoch_list,accuracy_list)
    plt.title("CNN  Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()