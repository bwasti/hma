import numpy as np
import hma

a_np = np.random.randn(2,2) #array([[1,2],[3,4]])
b_np = np.random.randn(2,2) #array([[1,3],[7,2]])

a_hma = hma.from_numpy(a_np)
b_hma = hma.from_numpy(b_np)
c_hma = hma.mul([a_hma, b_hma])[0]
print(hma.to_numpy(c_hma))

import torch

a_torch = torch.tensor(a_np)
b_torch = torch.tensor(b_np)
c_torch = torch.mul(a_torch, b_torch)
print(c_torch.numpy())

import time

for _ in range(100):
  c_hma = hma.mul([a_hma, b_hma])[0]
t = time.time()
for _ in range(100):
  c_hma = hma.mul([a_hma, b_hma])[0]
d = time.time() - t

print(f"{d*1e6 :.2f}")

for _ in range(100):
  c_torch = torch.mul(a_torch, b_torch)
t = time.time()
for _ in range(100):
  c_torch = torch.mul(a_torch, b_torch)
d = time.time() - t

print(f"{d*1e6 :.2f}")
