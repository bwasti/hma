import numpy as np
import hma
import pyhma as ph
import torch
import time

iters = 10000
a = np.random.randn(1,1)
a_ph = ph.Tensor(a)
a_t = torch.tensor(a)
h2 = ph.Tensor(np.array(2.0)).broadcast_like(a_ph)

for _ in range(iters):
  _ = (a_t + a_t) / 2
t = time.time()
for _ in range(iters):
  a_t = (a_t + a_t) / 2
print(a_t.numpy())
print("PT: ", time.time() - t)

t = time.time()
for _ in range(iters):
  a_ph = (a_ph + a_ph) / 2
print(a_ph.np())
print("PH: ", time.time() - t)

size = 128
iters = 20000

print()

def loss_fn(a, r, x):
  y_r = r * x
  y = a * x
  diff = y_r - y
  loss = (diff * diff).sum()
  loss = loss / float(size)
  return loss

a = np.random.randn(size).astype(np.float32)
ref = np.arange(size).astype(np.float32)
x = np.random.randn(size).astype(np.float32)

t = time.time()
for _ in range(iters):
  k = loss_fn(a, ref, x)
print("np: \t\t", time.time() - t)

x_ = torch.tensor(x)
a_ = torch.tensor(a)
r_ = torch.tensor(ref)

t = time.time()
for _ in range(iters):
  k = loss_fn(a_, r_, x_)
print("PT: no req_grad\t", time.time() - t)
a_.requires_grad = True
t = time.time()
for _ in range(iters):
  k = loss_fn(a_, r_, x_)
print("PT: req_grad\t", time.time() - t)

ph_x = ph.Tensor(x)
a = ph.Tensor(a)
r = ph.Tensor(ref)

t = time.time()
for _ in range(iters):
  k2 = loss_fn(a, r, ph_x)
print("hma: \t\t", time.time() - t)
assert np.allclose(k.detach().numpy(), k2.np())

print()

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
    assert np.allclose(a.np(), ref)
print("hma regression\t", time.time() - t)

device = torch.device('cuda', 0)
t = time.time()
a_ = torch.tensor(np.random.randn(size).astype(np.float32)).to(device)
r_ = r_.to(device)
a_ = a_.to(device)
a_.requires_grad = True
for _ in range(iters):
  x = torch.tensor(np.random.randn(size).astype(np.float32)).to(device)
  if a_.grad is not None:
    a_.grad.data = torch.zeros(a_.shape).to(device)
  y_r = r_ * x
  y = a_ * x
  diff = y_r - y
  loss = (diff * diff).sum()
  loss = loss / float(size)
  loss.backward()
  a_.data = a_ - a_.grad * 0.1
  if not ((_ + 1) % iters):
    #print(loss.item(), a_.detach().numpy())
    assert np.allclose(a_.detach().cpu().numpy(), ref)
print("PT regression gpu\t", time.time() - t)

#hma.debug(True)

a = np.random.randn(size).astype(np.float32)
a = ph.Tensor(a).cuda()
r = r.cuda()
t = time.time()
for _ in range(iters):
  x = ph.Tensor(np.random.randn(size).astype(np.float32)).cuda()
  y_r = r * x
  y = a * x
  diff = y_r - y
  loss = (diff * diff).sum()
  loss = loss / float(size)
  a_grad = loss.grad(a)()
  a = a - a_grad * 0.1
  if not ((_ + 1) % iters):
    #print(loss.np(), a.cpu().np())
    assert np.allclose(a.cpu().np(), ref)
print("hma regression gpu\t", time.time() - t)

