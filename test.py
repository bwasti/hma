import numpy as np
import hma

import torch

import time

a_np = np.random.randn(2,2) #array([[1,2],[3,4]])
b_np = np.random.randn(2,2) #array([[1,3],[7,2]])
a_inv_np = 1 / a_np

a_torch = torch.tensor(a_np)
a_inv_torch = torch.tensor(a_inv_np)
b_torch = torch.tensor(b_np)

a_hma_ = hma.from_numpy_(a_np)
a_inv_hma_ = hma.from_numpy_(a_inv_np)
b_hma_ = hma.from_numpy_(b_np)

a_hma = hma.from_numpy(a_np)
a_inv_hma = hma.from_numpy(a_inv_np)
b_hma = hma.from_numpy(b_np)

iters = 100000
c_hma_ = hma.mul_([a_hma_, b_hma_])[0]
t = time.time()
for _ in range(iters):
  c_hma_ = hma.mul_([a_hma_, c_hma_])[0]
  c_hma_ = hma.mul_([a_inv_hma_, c_hma_])[0]
print(time.time() - t, hma.to_numpy_(c_hma_))
c_torch = a_torch * b_torch
t = time.time()
for _ in range(iters):
	c_torch = a_torch * c_torch
	c_torch = a_inv_torch * c_torch
print(time.time() - t, c_torch)
print("---")
exit(0)

a_hma = hma.from_numpy_(a_np)
b_hma = hma.from_numpy_(b_np)
c_hma = hma.mul_([a_hma, b_hma])[0]
print(hma.to_numpy_(c_hma))

c_torch = torch.mul(a_torch, b_torch)
print(c_torch.numpy())

torch.testing.assert_allclose(c_torch.numpy(), hma.to_numpy_(c_hma))
print("passed correctness")

for _ in range(1000):
  c_hma = hma.mul_([a_hma, b_hma])[0]
t = time.time()
for _ in range(1000000):
  c_hma = hma.mul_([a_hma, b_hma])[0]
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

a_hma = hma.from_numpy(a_np)
b_hma = hma.from_numpy(b_np)
c_hma = hma.mul([a_hma, b_hma])[0]
a_hma_grad = hma.grad(c_hma, a_hma, c_hma)
print(a_hma_grad)
#hma.sum([c_hma])
