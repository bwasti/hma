import numpy as np
import hma
import pyhma as ph

a = np.random.randn(16)
a_ = ph.Tensor(a)
b = hma.to_cuda([a_.cTensor])[0]
c = hma.mul([b, b])[0]
d = hma.to_cpu([c])[0]
assert np.allclose(ph.Tensor(d).np(), a * a)

a = np.random.randn(16)
b = np.random.randn(16)
c_ref = a * b
a = ph.Tensor(a).cuda()
b = ph.Tensor(b).cuda()
c = a * b
assert np.allclose(c.cpu().np(), c_ref)

_ = np.random.randn(128)
k = ph.Tensor(_).cuda().sum()
print(k.cpu().np(), _.sum())
