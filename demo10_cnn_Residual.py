import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
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

#残差神经网络模型
class ResidualBlock(torch.nn.Module):
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.channels=channels
        self.conv1=torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2=torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)

    def forward(self,x):
        y=F.relu(self.conv1(x))
        y=self.conv2(y)
        return F.relu(x+y)


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet,self).__init__()
        self.conv1=torch.nn.Conv2d(1,16,kernel_size=5)
        self.conv2=torch.nn.Conv2d(16,32,kernel_size=5)

        self.mp=torch.nn.MaxPool2d(2)

        self.rblock1=ResidualBlock(channels=16)
        self.rblock2=ResidualBlock(channels=32)

        self.fc=torch.nn.Linear(512,10)

    def forward(self,x):
        in_size=x.size(0)
        x=self.mp(F.relu(self.conv1(x)))
        x=self.rblock1(x)
        x=self.mp(F.relu(self.conv2(x)))
        x=self.rblock2(x)
        x=x.view(in_size,-1)
        x=self.fc(x)
        return x

model=MnistNet()

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