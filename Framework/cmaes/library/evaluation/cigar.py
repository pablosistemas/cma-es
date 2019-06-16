import numpy as np

class Cigar:
  @staticmethod
  def run(x):
    if x.shape[0] < 2:
      raise 'Dimension must be greater than one'
    f = x[0] ** 2 + 1e6 * sum(np.square(x[1:]))
    return f
