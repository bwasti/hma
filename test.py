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

torch.testing.assert_allclose(c_torch.numpy(), hma.to_numpy(c_hma))
print("passed correctness")

import time

for _ in range(1000):
  c_hma = hma.mul([a_hma, b_hma])[0]
t = time.time()
for _ in range(1000000):
  c_hma = hma.mul([a_hma, b_hma])[0]
d1 = time.time() - t

print(f"{d1 :.2f}us per iter")

for _ in range(1000):
  c_torch = torch.mul(a_torch, b_torch)
t = time.time()
for _ in range(1000000):
  c_torch = torch.mul(a_torch, b_torch)
d2 = time.time() - t

print(f"{d2 :.2f}us per iter")

print(f"{(d2 - d1) / d2 * 100:.2f}% faster")
