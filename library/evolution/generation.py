import numpy as np

class Next:
  @staticmethod
  def run(N, lambda_pop_size, column_vector_xmean, sigma, B, D):
    std_norm_dist = np.array([np.random.normal(0, 1, N) for offs in range(lambda_pop_size)]).T
    mutation_dist = np.array([column_vector_xmean + sigma * B.dot(D.dot(std_norm_dist[:, k])) for k in range(lambda_pop_size)])
    return [std_norm_dist, mutation_dist]
