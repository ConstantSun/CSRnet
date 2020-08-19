import torch

# a = torch.arange(.5,1000.5,1.).reshape(100, 10).type(torch.float64)
# a = torch.arange(.5,1000.5,1.).reshape(100, 10)
# a = torch.arange(0,1000).reshape(100, 10).type(torch.float32)

a = torch.randn(100, 10)
def softmax(x): return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)
def nl(input, target): return -input[range(target.shape[0]), target].log().mean()
# # target: (100), input: (100,10)

b = torch.arange(100)

b /= 10

print("a dtype: ", a.dtype)
print("b dtype: ", b.dtype)
print("b shape: ", b.shape)

smf = softmax(a)
loss = nl(smf, b)
print("smf shape: ", smf.shape)
# print("smf : ", smf )
print(smf[range(100), b])
print(smf)

import numpy as np
mx = np.ones((100, 200), np.float32)
a = 0
for i in range(100):
    for j in range(200):
        

