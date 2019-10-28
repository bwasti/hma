import numpy as np
import hma
import pyhma as ph
import torch
import time

size = 64
iters = 10000
# y = a * x
a = np.random.randn(size).astype(np.float32)
ref = np.arange(size).astype(np.float32)

a_ = torch.tensor(a)
#a_.requires_grad = True
r_ = torch.tensor(ref)
a = ph.Tensor(a)
r = ph.Tensor(ref)

def loss_fn(a, r, x):
  y_r = r * x
  y = a * x
  diff = y_r - y
  loss = (diff * diff).sum()
  loss = loss / float(size)
  return loss

x = np.random.randn(size).astype(np.float32)
x_ = torch.tensor(x)
ph_x = ph.Tensor(x)
t = time.time()
for _ in range(iters):
  k = loss_fn(a_, r_, x_)
print("PT: no req_grad\t", time.time() - t)
a_.requires_grad = True
t = time.time()
for _ in range(iters):
  k = loss_fn(a_, r_, x_)
print("PT: req_grad\t", time.time() - t)
t = time.time()
for _ in range(iters):
  k2 = loss_fn(a, r, ph_x)
print("hma: \t\t", time.time() - t)
assert np.allclose(k.detach().numpy(), k2.np())

t = time.time()
for _ in range(iters):
  x = torch.tensor(np.random.randn(size).astype(np.float32))
  if a_.grad is not None:
    a_.grad.data = torch.zeros(a_.shape)
  y_r = r_ * x
  y = a_ * x
  diff = y_r - y
  loss = (diff * diff).sum()
  loss = loss / float(size)
  loss.backward()
  a_.data = a_ - a_.grad * 0.1
  if not ((_ + 1) % iters):
    #print(loss.item(), a_.detach().numpy())
    assert np.allclose(a_.detach().numpy(), ref)
print("PT regression\t", time.time() - t)

t = time.time()
for _ in range(iters):
  x = ph.Tensor(np.random.randn(size).astype(np.float32))
  y_r = r * x
  y = a * x
  diff = y_r - y
  loss = (diff * diff).sum()
  loss = loss / float(size)
  a_grad = loss.grad(a)()
  a = a - a_grad * 0.1
  if not ((_ + 1) % iters):
    #print(loss.np(), a.np())
    assert np.allclose(a_.detach().numpy(), ref)
print("hma regression\t", time.time() - t)

