import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
import math
import sys

sys.path.append(os.path.join(os.getcwd(), "out"))
import cuda_learn


M = 128
N = 128
K = 1024

dtype = torch.float32
a = torch.randn((M, K), dtype=dtype).cuda()
b = torch.randn((K, N), dtype=dtype).cuda()
result = torch.randn((M, N), dtype=dtype).cuda()

golden = a @ b

cuda_learn.matmul(a, b, result)

print(result)
print(golden)

diff = result.cpu() - golden.cpu()
print(diff.max(), diff.min())

allclose = torch.allclose(result.cpu(), golden.cpu(), rtol=1e-3, atol=1e-3)
print("cuda_learn.matmul: ", allclose)
