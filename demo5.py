import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# 用pytorch实现线性回归
#进给速度为6000，温降状态下的数据。
x_data = torch.Tensor([[0.0,100.0,6000.0],[10.0,100.0,6000.0],[20.0,100.0,6000.0],[30.0,100.0,6000.0],
                       [40.0, 100.0, 6000.0],[50.0,100.0,6000.0],[60.0,100.0,6000.0],[70.0,100.0,6000.0],[80.0,100.0,6000.0],[90.0,100.0,6000.0],
                       [0.0,200.0,6000.0],[10.0,200.0,6000.0],[20.0,200.0,6000.0],[30.0,200.0,6000.0],
                       [40.0, 200.0, 6000.0],[50.0,200.0,6000.0],[60.0,200.0,6000.0],[70.0,200.0,6000.0],[80.0,200.0,6000.0],[90.0,200.0,6000.0],
                       [0.0,300.0,6000.0],[10.0,300.0,6000.0],[20.0,300.0,6000.0],[30.0,300.0,6000.0],
                       [40.0, 300.0, 6000.0],[50.0,300.0,6000.0],[60.0,300.0,6000.0],[70.0,300.0,6000.0],[80.0,300.0,6000.0],[90.0,300.0,6000.0]
                       ])
y_data = torch.Tensor([[0.0132], [0.0085], [0.0052], [0.0035], [0.0024], [0.0014], [0.0009],[0.0005],[0.0002],[-0.0002],
                       [0.0272], [0.0189], [0.0126], [0.0086], [0.0057], [0.0037], [0.0023],[0.0012],[0.0007],[-0.0001],
                       [0.0419], [0.0291], [0.0203], [0.0144], [0.0103], [0.0071], [0.0049],[0.0033],[0.0023],[0.0012],
                       ])


# 使用MinMaxScaler进行归一化
scaler = MinMaxScaler()
x_data_normalized = scaler.fit_transform(x_data.numpy())
# 将归一化后的数据转换回Tensor
x_data_normalized = torch.Tensor(x_data_normalized)

class LinearModel(torch.nn.Module):  # module可以自动的识别计算图进行反馈
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc_1 = torch.nn.Linear(3, 10)
        self.fc_2= torch.nn.Linear(10,1)

    def forward(self, x):
        x=torch.relu(self.fc_1(x))
        y_pred = self.fc_2(x)
        return y_pred

model = LinearModel()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epoch_list = []
loss_list = []

num_epochs = 50000
for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
    y_pred = model.forward(x_data_normalized)
    loss = criterion(y_pred, y_data)
    epoch_list.append(epoch)
    loss_list.append(loss.tolist())
    # print(epoch, loss.tolist())
    optimizer.zero_grad()  # 梯度归零
    loss.backward()  # 进行反向传播
    optimizer.step()  # 根据学习率梯度自动进行更新（x，b）


print('w1=', model.fc_1.weight)
print('b1=', model.fc_1.bias)
print('w2=', model.fc_2.weight)
print('b2=', model.fc_2.bias)
print("loss:",loss_list[-1])
# 测试模型
x_test = torch.Tensor([[90, 300.0, 6000.0]])
# 对测试数据进行归一化
x_test_normalized = torch.Tensor(scaler.transform(x_test.numpy()))

y_test = model(x_test_normalized)
print('y_pred=', y_test.data)

# plt.plot(epoch_list, loss_list)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Rprop")
# plt.show()
