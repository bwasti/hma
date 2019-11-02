import hma
import pyhma as ph
import numpy as np
import torch

a = np.random.randn(128,128)
b = np.random.randn(128,128)
a = ph.Tensor(a)
b = ph.Tensor(b)
c = a * b
exit(0)

a = np.random.randn(128,128)
a = ph.Tensor(a)
print(str(a.shape))
j = hma.Size()
k = hma.Size()
v = ph.Tensor((j, k))
s = v.shape
print(s)
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