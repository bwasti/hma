import hma
import numpy as np

CUDA = hma.get_tag("CUDA")
CPU = hma.get_tag("CPU")

# Only stores hma.Tensor
class Tensor:

  def __init__(self, obj):
    if type(obj) is np.ndarray:
      self.cTensor = hma.from_numpy(obj)
    elif type(obj) is hma.Tensor:
      self.cTensor = obj
    elif type(obj) is type(self):
      self.cTensor = obj.cTensor
    else:
      raise Exception("Can't ingest obj", type(obj))

  def scalar_like(self, scalar):
    np_array = np.array(scalar)
    out = self.__class__(hma.from_numpy(np_array))
    return out.broadcast_like(self)

  def mul(self, other):
    if type(other) is not Tensor:
      other = self.scalar_like(other)
    return self.__class__(hma.mul([self.cTensor, other.cTensor])[0])

  def __mul__(self, other):
    return self.mul(other)

  def __rmul__(self, other):
    return self.mul(other)

  def div(self, other):
    if type(other) is not Tensor:
      other = self.scalar_like(other)
    return self.__class__(hma.div([self.cTensor, other.cTensor])[0])

  def __div__(self, other):
    return self.div(other)

  def __rdiv__(self, other):
    return self.div(other)
    
  def __truediv__(self, other):
    return self.div(other)

  def add(self, other):
    if type(other) is not Tensor:
      other = self.scalar_like(other)
    return self.__class__(hma.add([self.cTensor, other.cTensor])[0])

  def __add__(self, other):
    return self.add(other)

  def __radd__(self, other):
    return self.add(other)

  def sub(self, other):
    if type(other) is not Tensor:
      other = self.scalar_like(other)
    return self.__class__(hma.sub([self.cTensor, other.cTensor])[0])

  def __sub__(self, other):
    return self.sub(other)

  def __rsub__(self, other):
    return self.sub(other)

  def neg(self):
    return self.__class__(hma.neg([self.cTensor])[0])

  def __neg__(self):
    return self.neg()

  def sum(self):
    return self.__class__(hma.sum([self.cTensor])[0])

  def broadcast_like(self, like):
    if self.tag == CUDA and like.tag == CPU:
      self = self.cpu()
    if self.tag == CPU and like.tag == CUDA:
      self = self.cuda()
    return self.__class__(hma.broadcast([self.cTensor, like.cTensor])[0])

  def np(self):
    return hma.to_numpy(self.cTensor)

  def cuda(self):
    return self.__class__(hma.to_cuda([self.cTensor])[0])

  def cpu(self):
    return self.__class__(hma.to_cpu([self.cTensor])[0])

  @property
  def tag(self):
    return hma.get_tag(self.cTensor)

  def grad(self, other):
    def g(j=None):
      if j is None:
        j = Tensor(np.array(1)).broadcast_like(self)
      return self.__class__(hma.grad(self.cTensor, other.cTensor, j.cTensor))
    return g
