import torchvision
train_set=torchvision.datasets.MNIST(root='E:\项目\python\dataset',train=True,download=False)
test_set=torchvision.datasets.MNIST(root='E:\项目\python\dataset',train=False,download=False)