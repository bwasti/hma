import numpy as np
import hma
import pyhma as ph
import torch

a_ = np.random.randn(128,128).astype(np.float32)
b_ = np.random.randn(128,128).astype(np.float32)
ones = np.ones((128,128)).astype(np.float32)

def f0(a, b):
  return a + b
def f1(a, b):
  return a * b
def f2(a, b):
  return a * b + a
def f3(a, b):
  return (a * b + a) * a + b
def f4(a, b):
  c = a + b
  for _ in range(100):
    c = a * c
  return c

funcs = [ f0, f1, f2, f3, f4 ]

for func in funcs:
  a = ph.Tensor(a_)
  b = ph.Tensor(b_)
  c = func(a, b)
  g = c.grad(a)
  a_grad = g(ph.Tensor(ones)).np()

  a = torch.tensor(a_)
  a.requires_grad = True
  b = torch.tensor(b_)
  c = func(a,b)
  c.backward(torch.tensor(ones))

  torch.testing.assert_allclose(a_grad, a.grad.numpy())

for _ in range(4):
  a_np = np.random.randn(_,_).astype(np.float32)
  b_np = np.random.randn(_,_).astype(np.float32)

  a = hma.from_numpy(a_np)
  b = hma.from_numpy(b_np)

  c = hma.mul([a, b])[0]
  c = hma.mul([a, c])[0]
  c = hma.mul([a, c])[0]
  j = hma.from_numpy(np.ones((_,_)))
  a_grad = hma.grad(c, a, j)

  a_ = torch.tensor(a_np)
  a_.requires_grad = True
  b_ = torch.tensor(b_np)

  c_ = a_ * b_
  c_ = a_ * c_
  c_ = a_ * c_
  j_ = torch.tensor(np.ones((_,_)))
  c_.backward(j_)

  torch.testing.assert_allclose(hma.to_numpy(a_grad), a_.grad)
