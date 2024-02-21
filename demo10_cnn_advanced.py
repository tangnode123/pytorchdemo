import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
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
        y=self.conv2
        return F.relu(x+y)


#GooleNet
class InceptionA(torch.nn.Module):
    def __init__(self,in_channels):
        super(InceptionA,self).__init__()
        self.branch1x1=torch.nn.Conv2d(in_channels,16,kernel_size=1)

        self.branch5x5_1=torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_2=torch.nn.Conv2d(16,24,kernel_size=5,padding=2)

        self.branch3x3_1=torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2=torch.nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3=torch.nn.Conv2d(24,24,kernel_size=3,padding=1)

        self.branch_pool=torch.nn.Conv2d(in_channels,24,kernel_size=1)

    def forward(self,x):
        branch1x1=self.branch1x1(x)

        branch5x5=self.branch5x5_1(x)
        branch5x5=self.branch5x5_2(branch5x5)

        branch3x3=self.branch3x3_1(x)
        branch3x3=self.branch3x3_2(branch3x3)
        branch3x3=self.branch3x3_3(branch3x3)

        branch_pool=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool=self.branch_pool(branch_pool)

        outputs=[branch1x1,branch5x5,branch3x3,branch_pool]#88个通道数
        return torch.cat(outputs,dim=1)

class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet,self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(88,20,kernel_size=5)

        self.incep1=InceptionA(in_channels=10)
        self.incep2=InceptionA(in_channels=20)

        self.mp=torch.nn.MaxPool2d(2)
        self.fc=torch.nn.Linear(1408,10)

    def forward(self,x):
        in_size=x.size(0)
        x=F.relu(self.mp(self.conv1(x)))
        x=self.incep1(x)
        x=F.relu(self.mp(self.conv2(x)))
        x=self.incep2(x)
        x=x.view(in_size,-1)
        x=self.fc(x)
        return x

model=MnistNet()

device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
model.to(device)

criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def MnistTrain(epoch,num_epochs):
    losses=[]
    accuracy=[]
    loop=tqdm((train_loader),total=len(train_loader))
    for data,targets in loop:
        inputs,targets=data.to(device),targets.to(device)
        optimizer.zero_grad()

        outputs=model(inputs)
        loss=criterion(outputs,targets)

        losses.append(loss)
        loss.backward()
        _,predictions=outputs.max(1)
        num_correct=(predictions==targets).sum()
        running_train_acc=float(num_correct)/float(inputs.shape[0])
        accuracy.append(running_train_acc)
        optimizer.step()
        loop.set_description(f'Train Epoch [{epoch}/{num_epochs}]')
        loop.set_postfix(loss=loss.item(), acc=running_train_acc)

def MnistTest(epoch, num_epochs):
    total = 0
    correct = 0
    accuracy = []
    loop = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for data, targets in loop:
            images, labels = data.to(device), targets.to(device)
            outputs = model(images)
            _, predictions = outputs.max(1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            running_test_acc = correct / total
            accuracy.append(running_test_acc)
            loop.set_description(f'Test Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(acc=running_test_acc)

epoch_list=[]
accuracy_list=[]
#提取特征的思考FFT
if __name__=='__main__':
    for epoch in range(10):
        MnistTrain(epoch,10)
        epoch_list.append(epoch)
        MnistTest(epoch,10)
    # plt.plot(epoch_list,accuracy_list)
    # plt.title("CNN  Accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.show()