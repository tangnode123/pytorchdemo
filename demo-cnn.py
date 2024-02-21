#降低运算需求，减少元素数量，变为多通道
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
batch_size=64
# 取到的图像变为一个1x28x28的张量介于0到1之间
transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))#均值，标准差
])
train_dataset=datasets.MNIST(root='./dataset/mnist/',train=True,download=True,transform=transforms)
train_loader=DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
test_dataset=datasets.MNIST(root='./dataset/mnist/',train=False,download=True,transform=transforms)
test_loader=DataLoader(test_dataset,shuffle=False,batch_size=batch_size)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5)
        self.conv3=torch.nn.Conv2d(20,30,kernel_size=2)
        self.pooling=torch.nn.MaxPool2d(2)
        self.pooling2=torch.nn.MaxPool2d(3)
        self.fc1=torch.nn.Linear(30,20)
        self.fc2=torch.nn.Linear(20,10)
    def forward(self,x):
        batch_size=x.size(0)
        x=F.relu(self.pooling(self.conv1(x)))
        x=F.relu(self.pooling(self.conv2(x)))
        x=F.relu(self.pooling2(self.conv3(x)))
        x=x.view(batch_size,-1)
        x=self.fc1(x)
        x=self.fc2(x)
        return x
model=Net()
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.cuda()
criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)#冲量冲破局部最小值
def train(epoch):
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
        if(batch_idx%300==299):
            print('[%d,%5d] loss:%.3f'%(epoch+1,batch_idx+1,running_loss/300))
            running_loss=0.0
def atest():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            _,predicted=torch.max(outputs.data,dim=1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print('Accuracy on test set:%d %%'%(100*correct/total))
if __name__=="__main__":
    for epoch in range(10):
        train(epoch)
        atest()

# in_channels,out_channels=5,10
# width,height=100,100
# kernel_size=3
# batch_size = 1
# input=torch.randn(batch_size,in_channels,width,height)
# conv_layer=torch.nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size)
# output=conv_layer(input)
# print(input.shape)
# print(output.shape)
# print(conv_layer.weight.shape)
#padding填充一圈零，stride每次跳两格
# input=[
#     3,4,6,5,7,
#     2,4,6,8,2,
#     1,6,7,8,4,
#     9,7,4,6,2,
#     3,7,5,4,1
# ]
# input=torch.Tensor(input).view(1,1,5,5)
# conv_layer=torch.nn.Conv2d(1,1,kernel_size=3,padding=1,stride=2,bias=False)
#
# kernel=torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3)
# conv_layer.weight.data=kernel.data
#
# output=conv_layer(input)
# print(output)
#maxpooling最大池化层
# input=[
#     3,4,6,5,
#     2,4,6,8,
#     1,6,7,8,
#     9,7,4,6
# ]
# input=torch.Tensor(input).view(1,1,4,4)
# maxpooling_layer=torch.nn.MaxPool2d(kernel_size=2)
# output=maxpooling_layer(input)
# print(output)
