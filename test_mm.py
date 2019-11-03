import hma
import pyhma as ph
import numpy as np

hma.set_debug(True)

for A in [2,5,16,27]:
	for B in [2,5,16,27]:
		for C in [2,5,16,27]:
			anp = np.random.randn(A, B).astype(np.float32)
			bnp = np.random.randn(B, C).astype(np.float32)
			a = ph.Tensor(anp).cuda()
			b = ph.Tensor(bnp).cuda()
			c = ph.Tensor(hma.mm_nn([a.cTensor, b.cTensor])[0])

			torch.testing.assert_allclose(c.cpu().np(), (anp) @ (bnp))

			anp = np.random.randn(A, B).astype(np.float32)
			bnp = np.random.randn(C, B).astype(np.float32)
			a = ph.Tensor(anp).cuda()
			b = ph.Tensor(bnp).cuda()
			c = ph.Tensor(hma.mm_nt([a.cTensor, b.cTensor])[0])

			torch.testing.assert_allclose(c.cpu().np(), (anp) @ (bnp.T))

			anp = np.random.randn(B, A).astype(np.float32)
			bnp = np.random.randn(B, C).astype(np.float32)
			a = ph.Tensor(anp).cuda()
			b = ph.Tensor(bnp).cuda()
			c = ph.Tensor(hma.mm_tn([a.cTensor, b.cTensor])[0])

			torch.testing.assert_allclose(c.cpu().np(), (anp.T) @ (bnp))

			anp = np.random.randn(B, A).astype(np.float32)
			bnp = np.random.randn(C, B).astype(np.float32)
			a = ph.Tensor(anp).cuda()
			b = ph.Tensor(bnp).cuda()
			c = ph.Tensor(hma.mm_tt([a.cTensor, b.cTensor])[0])

			torch.testing.assert_allclose(c.cpu().np(), (anp.T) @ (bnp.T))
