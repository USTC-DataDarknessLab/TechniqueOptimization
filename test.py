import torch
import torch.nn as nn
import torch.optim as optim
import copy

# 模拟数据集
torch.manual_seed(0)
X = torch.randn(128, 10)
y = torch.randn(128, 1)

# 模型定义
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def train_single_group(model, data_loader, optimizer, loss_fn, grad_accum_steps):
    model.train()
    optimizer.zero_grad()
    for i, (x_batch, y_batch) in enumerate(data_loader):
        output = model(x_batch)
        # print(x_batch.shape, y_batch.shape, output.shape)  # 打印批次形状
        loss = loss_fn(output, y_batch)
        loss.backward()
        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

def train_multi_group(model, data_loader, optimizer, loss_fn, grad_accum_steps, group_size):
    model.train()
    total_batches = len(data_loader)
    group_batches = total_batches // group_size
    for g in range(group_size):
        optimizer.zero_grad()
        for i in range(g * group_batches, (g + 1) * group_batches):
            x_batch, y_batch = data_loader[i]
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
        optimizer.step()

# 构造 DataLoader（不使用随机顺序）
def get_batches(X, y, batch_size):
    return [(X[i:i+batch_size], y[i:i+batch_size]) for i in range(0, len(X), batch_size)]

# 参数设定
batch_size = 8
grad_accum_steps = 4
group_size = 4
num_epochs = 1

# 生成批次数据
batches = get_batches(X, y, batch_size)

# 初始化模型
model1 = SimpleModel()
model2 = copy.deepcopy(model1)

optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 训练
import time
t0=time.time()
train_single_group(model1, batches, optimizer1, loss_fn, grad_accum_steps)
t1=time.time()
train_multi_group(model2, batches, optimizer2, loss_fn, grad_accum_steps, group_size)
t2=time.time()
print(f"单组训练时间: {t1 - t0:.4f}秒")
print(f"多组训练时间: {t2 - t1:.4f}秒")

# 对比参数是否一致
def compare_models(m1, m2):
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        if not torch.allclose(p1, p2, atol=1e-6):
            print("模型参数不一致")
            return
    print("模型参数完全一致")

compare_models(model1, model2)
