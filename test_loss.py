import torch 
import torch.nn.functional as F 

x = torch.randn((10, 4), dtype=torch.float32)
# print(x)
target = torch.randint(low=0, high=4, size=(10,), dtype=torch.long)
# print(target)
y = torch.zeros_like(x, dtype=torch.float32)
y[[range(target.shape[0])], target] = 1
# print(y)

x_ = torch.sigmoid(x)
# print(x_)
cre1 = torch.nn.BCELoss()
loss1 = cre1(x_, y)
cre2 = torch.nn.BCEWithLogitsLoss()
loss2 = cre2(x, y)
print("loss1 : %f" % (loss1.item()))
print("loss2 : %f" % (loss2.item()))

x_log = torch.log(x_)
x_log_1 = torch.log(1-x_)
loss3_1 = torch.mean(y*x_log*-1)
loss3_2 = torch.mean((1-y)*x_log_1*-1)
print("loss3 : %f" % (loss3_1.item() + loss3_2.item()))