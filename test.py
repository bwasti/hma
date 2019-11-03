import hma
import pyhma as ph
import numpy as np
import torch
import time

hma.set_debug(True)
device = torch.device('cuda', 0)

N = 64
C_i = 128
C_o = 32
xnp = np.random.randn(N, C_i).astype(np.float32)
wnp = np.random.randn(C_o, C_i).astype(np.float32)
snp = np.random.randn(N, C_o).astype(np.float32)

xt = torch.tensor(xnp).to(device)
wt = torch.tensor(wnp).to(device)
st = torch.tensor(snp).to(device)

yt = torch.nn.functional.linear(xt, wt)

x = ph.Tensor(xnp).cuda()
w = ph.Tensor(wnp).cuda()
s = ph.Tensor(snp).cuda()

y = ph.Tensor(hma.mm_nt([x.cTensor, w.cTensor])[0])

ynp = y.cpu().np()
torch.testing.assert_allclose(y.cpu().np(), yt.cpu())

print("passed.")

t = time.perf_counter()
for i in range(10000):
	_ = torch.nn.functional.linear(xt, wt)
print(_.cpu().numpy().shape)
print(time.perf_counter() - t)

t = time.perf_counter()
for i in range(10000):
	_ = ph.Tensor(hma.mm_nt([x.cTensor, w.cTensor])[0])
print(_.cpu().np().shape)
print(time.perf_counter() - t)

xt.requires_grad = True
wt.requires_grad = True

yt = torch.nn.functional.linear(xt, wt) * st
y = ph.Tensor(hma.mm_nt([x.cTensor, w.cTensor])[0]) * s
torch.testing.assert_allclose(y.cpu().np(), yt.cpu())

yt.sum().backward()

w_grad = y.sum().grad(w)()
torch.testing.assert_allclose(w_grad.cpu().np(), wt.grad.cpu())
print("passed dw")

x_grad = y.sum().grad(x)()
torch.testing.assert_allclose(x_grad.cpu().np(), xt.grad.cpu())
print("passed dx")

print("passed grad.")

xnp = np.random.randn(4,3,128,128).astype(np.float32)
wnp = np.random.randn(5,3,3,3).astype(np.float32)

x = ph.Tensor(xnp).cuda()
w = ph.Tensor(wnp).cuda()

hma.resolve(x.cTensor)
hma.resolve(w.cTensor)

y = ph.Tensor(hma.conv([x.cTensor, w.cTensor])[0])

t = time.perf_counter()
for i in range(1000):
  _ = ph.Tensor(hma.conv([x.cTensor, w.cTensor])[0])
print(_.cpu().np().shape)
print(time.perf_counter() - t)

onp = np.ones((4,5,126,126))
o = ph.Tensor(onp).cuda()
dx = y.sum().grad(x)()
dw = y.sum().grad(w)()

xt = torch.tensor(xnp)
wt = torch.tensor(wnp)

t = time.perf_counter()
for i in range(1000):
  _ = torch.nn.functional.conv2d(xt, wt)
print(_.cpu().numpy().shape)
print(time.perf_counter() - t)

xt.requires_grad = True
wt.requires_grad = True

t = time.perf_counter()
for i in range(1000):
  _ = torch.nn.functional.conv2d(xt, wt)
print(_.cpu().detach().numpy().shape)
print(time.perf_counter() - t)

yt = torch.nn.functional.conv2d(xt, wt)


yt.sum().backward()

torch.testing.assert_allclose(y.cpu().np(), yt      , rtol=0.001, atol=0.001)
torch.testing.assert_allclose(dx.cpu().np(), xt.grad, rtol=0.001, atol=0.001)
torch.testing.assert_allclose(dw.cpu().np(), wt.grad, rtol=0.001, atol=0.001)

a = np.random.randn(128,128)
b = np.random.randn(128,128)
a = ph.Tensor(a)
b = ph.Tensor(b)
c = a * b

a = np.random.randn(128,128)
a = ph.Tensor(a)
print(a.shape)
j = hma.Size()
k = hma.Size()
v = ph.Tensor((j, k))
s = v.shape
print(s)
hma.set_debug(False)
# TODO when debug is on this fails
assert int(v.sum().shape[0]) == 1
 
def test_bin_op(f):
  for s in [(1,), (2,2), (128,128)]:
    an = np.random.randn(*s)
    bn = np.random.randn(*s)
    ap = ph.Tensor(an).cuda()
    bp = ph.Tensor(bn).cuda()
    cp = f(ap, bp).cpu().np()
    torch.testing.assert_allclose(cp, f(an, bn))

def test_unary_op(f, *args):
  for s in [(1,), (2,2), (128,128)]:
    an = np.random.randn(*s)
    ap = ph.Tensor(an).cuda()
    cp = f(ap, *args).cpu().np()
    torch.testing.assert_allclose(cp, f(an))

def test_sum():
  for s in [(1,), (2,2), (128,128)]:
    an = np.random.randn(*s)
    ap = ph.Tensor(an).cuda()
    cp = ap.sum().cpu().np()
    torch.testing.assert_allclose(cp, an.sum())

def test_broadcast():
  for s in [(1,), (2,2), (128,128)]:
    an = np.random.randn(1)
    liken = np.random.randn(*s)
    ap = ph.Tensor(an).cuda()
    likep = ph.Tensor(liken).cuda()
    cp = ap.broadcast_like(likep).cpu().np()
    torch.testing.assert_allclose(cp, np.broadcast_to(an, s))

import operator 
test_bin_op(operator.add)
test_bin_op(operator.sub)
test_bin_op(operator.truediv)
test_bin_op(operator.mul)
test_unary_op(operator.neg)
test_sum()
test_broadcast()
