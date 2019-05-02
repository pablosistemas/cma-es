import math
import numpy as np

class Cmaes:
  def __init__(self, dimension_size, recombination):
    self.recombination = recombination

    # Set dimension, fitness fct, stop criteria, start values...
    self.N = dimension_size
    self.xmean = np.random.uniform(0, 1, self.N).reshape((self.N, 1))
    self.zmean = None
    self.sigma = 0.5

    # Stop
    self.stopeval = 1e3 * self.N ** 2

    # Selection
    self.lambda_pop_size = 4 + math.floor(3 * np.log(self.N))
    self.mu_rank_elite_size = math.floor(self.lambda_pop_size / 2.0)
    self.weights = np.log(self.mu_rank_elite_size + 1.0 / 2) \
        - np.array([np.log(i + 1) for i in range(self.mu_rank_elite_size)])

    self.weights = (self.weights / sum(self.weights)).reshape((self.mu_rank_elite_size, 1))
    self.mueff = np.asscalar((sum(self.weights) ** 2 ) / sum(np.square(self.weights)))

    # Adaptation
    # # Step size control
    self.cumulate_sigma_control = (self.mueff + 2) / (self.N + self.mueff + 5)
    self.damps = 1 + 2.0 * max(0, np.sqrt((self.mueff - 1) / (self.N + 1)) - 1) + self.cumulate_sigma_control
    # # Covariance matrix
    self.cov_adapt_time_constant = (4 + self.mueff / self.N) / (self.N + 4.0 + 2.0 * self.mueff / self.N)
    self.alpha_cov = 2.0
    self.c1 = self.alpha_cov / ((self.N + 1.3) ** 2 + self.mueff)
    self.cmu = min(
      1.0 - self.c1,
      self.alpha_cov * (self.mueff - 2. + 1.0 / self.mueff) / ((self.N + 2) ** 2 + self.alpha_cov * self.mueff / 2.0)
    )

    # Dynamic (internal) strategy parameters and constants
    # Page 16, Evolution Strategies paper, section 5
    self.pc = np.zeros((self.N, 1)) # weighted mean selected mutation step
    self.ps = np.zeros((self.N, 1))
    self.B = np.identity(self.N)
    self.D = np.identity(self.N)
    self.C = np.identity(self.N) # self.B.dot(self.D.dot(np.matrix.transpose(self.B.dot(self.D))))
    self.chiN = self.N ** 0.5 * (1 - 1.0 / (4 * self.N) + 1.0 / (21 * self.N ** 2))

  def next(self, fn_eval, counteval, eigenval):
    std_norm_dist = np.array([np.random.normal(0, 1, self.N) for offs in range(self.lambda_pop_size)]).T
    mutation_dist = np.array([(self.xmean + self.sigma * self.B.dot(self.D.dot(std_norm_dist[:, k].reshape((self.N, 1))))).reshape(self.N) for k in range(self.lambda_pop_size)]).T

    offspring_evaluation = np.array([fn_eval(mutation_dist[:, offs]) for offs in range(self.lambda_pop_size)]).reshape((self.N, 1))
    counteval = counteval + self.lambda_pop_size

    idx_rank_offspring = self.__rank_sort(offspring_evaluation)

    self.xmean = self.recombination(
      mutation_dist[:, idx_rank_offspring[0: self.mu_rank_elite_size].reshape(self.mu_rank_elite_size)],
      self.weights
    )

    self.zmean = self.recombination(
      std_norm_dist[:, idx_rank_offspring[0: self.mu_rank_elite_size].reshape(self.mu_rank_elite_size)],
      self.weights
    )

    # print('zmean')
    # print(self.zmean)

    # print('xmean')
    # print(self.xmean)

    self.ps = (1 - self.cumulate_sigma_control) * self.ps + \
        (np.sqrt(self.cumulate_sigma_control * (2 - self.cumulate_sigma_control) * self.mueff)) * (self.B.dot(self.zmean))

    # print('ps:')
    # print(self.ps)

    self.hsig = np.linalg.norm(self.ps) / \
        np.sqrt(1 - (1 - self.cumulate_sigma_control) ** (2 * counteval / self.lambda_pop_size)) / \
        self.chiN < (1.4 + 2 / (self.N + 1))

    # print('hsig: ')
    # print(self.hsig)

    self.pc = (1 - self.cov_adapt_time_constant) * self.pc + \
        self.hsig * np.sqrt(self.cov_adapt_time_constant * (2 - self.cov_adapt_time_constant) * self.mueff) * (self.B.dot(self.D.dot(self.zmean.reshape((self.N, 1)))))

    BDzk = (self.B.dot(self.D.dot(std_norm_dist[:, idx_rank_offspring[0: self.mu_rank_elite_size].reshape(self.mu_rank_elite_size)])))
    self.C = (1 - self.c1 - self.cmu) * self.C \
      + self.c1 * (self.pc.dot(self.pc.T) \
          + (1 - self.hsig) * self.cov_adapt_time_constant * (2 - self.cov_adapt_time_constant) * self.C) \
          + self.cmu \
            * BDzk.dot(np.diag(self.weights.reshape(self.mu_rank_elite_size)).dot(BDzk.T))

    self.sigma = self.sigma * np.exp((self.cumulate_sigma_control / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

    if np.mod(counteval / self.lambda_pop_size, self.N / 10.0) < 1.0:
      eigenval = counteval
      self.C = np.triu(self.C) + np.triu(self.C, 1).T
      [self.D, self.B] = np.linalg.eig(self.C)
      self.D = np.diag(np.sqrt(self.D))

    # print('D:')
    # print(self.D)

    # print('C: ')
    # print(self.C)

    if offspring_evaluation[idx_rank_offspring[0]] == offspring_evaluation[idx_rank_offspring[math.floor(np.ceil(0.7 * self.lambda_pop_size))]]:
      self.sigma = self.sigma * np.exp(0.2 + self.cumulate_sigma_control / self.damps)

    # print('sigma')
    # print(self.sigma)

    return [std_norm_dist, mutation_dist, idx_rank_offspring, offspring_evaluation[idx_rank_offspring[0]], counteval, eigenval]

  def __rank_sort(self, evaluation):
    return np.argsort(evaluation.reshape(self.N))