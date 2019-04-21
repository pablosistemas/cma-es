import math
import numpy as np

class Cmaes:
  def __init__(self, dimension_size):
    self.N = dimension_size
    
    self.xmean = np.random.uniform(0, 1, self.N)
    self.sigma = 0.5
    self.stopfitness = 1e-10
    self.stopeval = 1e3 * self.N ** 2

    self.lambda_pop_size = math.floor(4 + np.floor(3 * np.log(self.N)))
    self.mu_rank_elite_size = math.floor(self.lambda_pop_size / 2.0)

    self.weights_selection = np.log(self.mu_rank_elite_size + 0.5) - np.array([np.log(i + 1) for i in range(self.mu_rank_elite_size)])

    self.weights_selection_sum = sum(self.weights_selection)
    self.weights = self.weights_selection / self.weights_selection_sum

    self.mueff = ( self.weights_selection_sum ** 2 ) / sum([weight ** 2 for weight in self.weights])

    self.cumulate_sigma_control = (self.mueff + 2) / (self.N + self.mueff + 5)

    self.cov_adapt_time_constant = (4 + self.mueff / self.N) / (self.N + 4.0 + 2.0 * self.mueff / self.N)
    self.c1 = 2.0 / ((self.N + 1.3) ** 2 + self.mueff)
    self.cmu = 2 * (self.mueff - 2. + 1 / self.mueff) / ((self.N + 2) ** 2 + 2 * self.mueff / 2)

    self.damps = 1 + 2.0 * np.max((0, np.sqrt((self.mueff - 1) / (self.N + 1)) - 1)) + self.cumulate_sigma_control

    self.pc = np.zeros((self.N, 1))
    self.ps = np.zeros((self.N, 1))
    self.B = np.identity(self.N)
    self.D = np.identity(self.N)

    self.C = self.B.dot(self.D.dot(np.matrix.transpose(self.B.dot(self.D))))

    self.chiN = self.N ** 0.5 * (1 - 1.0 / (4 * self.N) + 1.0 / (21 * self.N ** 2))
