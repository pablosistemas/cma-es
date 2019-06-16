import numpy as np

class Felli:
  @staticmethod
  def run(x):
    N = x.shape[0]
    if N < 2:
      raise 'Dimension must be greater than one'
    f = np.array([(1e6 * (i / (N - 1))) for i in range(N)]).dot(np.square(x))
    return f
