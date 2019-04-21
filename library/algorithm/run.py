import numpy as np
import math

class Run:
  def __init__(self, model, func, generation, recombination):
    self.model = model
    self.fn_eval = func
    self.recombination = recombination
    self.next_generation = generation

  def run(self):
    std_offspring = None
    mut_offspring = None
    idx_rank_offspring = None
    offspring_evaluation = None
    eigenval = 0
    counteval = 0
    model = self.model

    while(not self.__stop_condition(counteval, self.model.stopeval)):

      std_offspring, mut_offspring = self.next_generation(
        model.N, 
        model.lambda_pop_size,
        model.xmean,
        model.sigma,
        model.B,
        model.D
      )

      offspring_evaluation = [self.fn_eval(model.N, mut_offspring[:, offs]) for offs in range(model.lambda_pop_size)]
      counteval = counteval + model.lambda_pop_size

      idx_rank_offspring = self.__rank_sort(offspring_evaluation)

      xmean = self.recombination(
        mut_offspring[:, idx_rank_offspring[0: model.mu_rank_elite_size]],
        model.weights,
        model.mu_rank_elite_size
      )

      zmean = self.recombination(
        std_offspring[:, idx_rank_offspring[0: model.mu_rank_elite_size]],
        model.weights,
        model.mu_rank_elite_size
      )

      model.ps = (1 - model.cumulate_sigma_control) * model.ps + \
          (np.sqrt(model.cumulate_sigma_control * (2 - model.cumulate_sigma_control) * model.mueff)) * (model.B.dot(zmean))

      model.hsig = np.linalg.norm(model.ps) / \
          np.sqrt(1 - (1 - model.cumulate_sigma_control) ** (2 * counteval / model.lambda_pop_size)) / \
          model.chiN < 1.4 + 2/(model.N + 1)

      model.pc = (1 - model.cov_adapt_time_constant) * model.pc + \
          model.hsig * np.sqrt(model.cov_adapt_time_constant * (2 - model.cov_adapt_time_constant) * model.mueff) * (model.B.dot(model.D.dot(zmean)))

      model.C = (1 - model.c1 - model.cmu) * model.C + model.c1
      model.C = model.C.dot(model.pc.dot(model.pc.T) + (1 - model.hsig) * model.cov_adapt_time_constant * (2 - model.cov_adapt_time_constant) * model.C)
      model.BDOffs = model.B.dot(model.D.dot(std_offspring[:, idx_rank_offspring[0: model.mu_rank_elite_size]]))
      model.C = model.C + model.cmu * (model.BDOffs.dot(np.diag(model.weights)).dot(model.BDOffs.T))

      model.sigma = model.sigma * np.exp((model.cumulate_sigma_control / model.damps) * (np.linalg.norm(model.ps) / model.chiN - 1))

      if (counteval - eigenval) > (model.lambda_pop_size / (1 + model.cmu) / model.N / 10):
        eigenval = counteval
        model.C = np.triu(model.C) + np.triu(model.C, 1).T
        [model.D, model.B] = np.linalg.eig(model.C)
        model.D = np.diag(np.sqrt(model.D))

      if offspring_evaluation[0] <= model.stopfitness:
        break

      if offspring_evaluation[0] == offspring_evaluation[math.floor(np.ceil(0.7 * model.lambda_pop_size))]:
        model.sigma = model.sigma * np.exp(0.2 * model.cumulate_sigma_control / model.damps)

    xmin = mut_offspring[:, idx_rank_offspring[0]]

    return [xmin, counteval]

  def __stop_condition(self, counteval, stopeval):
    return counteval >= stopeval
  
  def __rank_sort(self, evaluation):
    return np.argsort(evaluation)
