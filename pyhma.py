import hma
import numpy as np

# Only stores hma.Tensor
class Tensor:

  def __init__(self, obj):
    if type(obj) is np.ndarray:
      self.cTensor = hma.from_numpy(obj)
    if type(obj) is hma.Tensor:
      self.cTensor = obj
    if type(obj) is type(self):
      self.cTensor = obj.cTensor

  def mul(self, other):
    return self.__class__(hma.mul([self.cTensor, other.cTensor])[0])

  def __mul__(self, other):
    return self.mul(other)

  def __rmul__(self, other):
    return self.mul(other)

  def add(self, other):
    return self.__class__(hma.add([self.cTensor, other.cTensor])[0])

  def __add__(self, other):
    return self.add(other)

  def __radd__(self, other):
    return self.add(other)

  def np(self):
    return hma.to_numpy(self.cTensor)

  def grad(self, other):
    def g(j):
      return self.__class__(hma.grad(self.cTensor, other.cTensor, j.cTensor))
    return g
