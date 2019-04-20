import numpy as np
import math

def rank_sort(evaluation):
  return np.argsort(evaluation)

def felli():
  pass

def main():
  
  N = 10
  xmean = np.random.uniform(0, 1, N)
  sigma = 0.5
  stopfitness = 1 ** (-10)
  stopeval = 1 ** (10) * N ** 2

  lambda_pop_size = math.floor(4 + np.floor(3 * np.log(N)))
  mu_rank_elite_size = math.floor(lambda_pop_size / 2.0)

  weights_selection = np.log(mu_rank_elite_size + 0.5) - np.array([np.log(i + 1) for i in range(mu_rank_elite_size)])

  weights_selection_sum = sum(weights_selection)
  weights = weights_selection / weights_selection_sum

  mueff = ( weights_selection_sum ** 2 ) / sum([weight ** 2 for weight in weights])

  cumulate_sigma_control = (mueff + 2) / (N + mueff + 5)

  cov_adapt_time_constant = (4 + mueff / N) / (N + 4.0 + 2.0 * mueff / N)
  c1 = 2.0 / ((N + 1.3) ** 2 + mueff)
  cmu = 2 * (mueff - 2. + 1 / mueff) / ((N + 2) ** 2 + 2 * mueff / 2)

  damps = 1 + 2.0 * np.max((0, np.sqrt((mueff - 1) / (N + 1)) - 1)) + cumulate_sigma_control

  pc = np.zeros((N, 1))
  ps = np.zeros((N, 1))
  B = np.identity(N)
  D = np.identity(N)

  C = B.dot(D.dot(np.matrix.transpose(B.dot(D))))

  eigenval = 0

  chiN = N ** 0.5 * (1 - 1.0 / (4 * N) + 1.0 / (21 * N ** 2))

  counteval = 0
  idx_rank_offspring = None
  offspring_evaluation = None

  def update_parameters():
    pass

  def stop_condition():
    return counteval >= stopeval

  def next_generation():
    std_norm_dist = np.array([np.random.normal(0, 1, N) for offs in range(lambda_pop_size)]).T
    mutation_dist = np.array([xmean + sigma * B.dot(D.dot(std_norm_dist[:,k])) for k in range(lambda_pop_size)])
    return [std_norm_dist, mutation_dist]

  def evaluate_objective_function(x):
    if x.shape[0] < 2:
      raise 'Dimension must be greater than one'
    f = np.array([(10 ** 6 * (i / (N - 1))) for i in range(N)]).dot(np.square(x))
    return f

  def recombination(offs):
    return offs.dot(weights.reshape(mu_rank_elite_size))


  while(not stop_condition()):

    std_offspring, mut_offspring = next_generation()
    offspring_evaluation = [evaluate_objective_function(mut_offspring[:,offs]) for offs in range(lambda_pop_size)]
    counteval = counteval + lambda_pop_size

    idx_rank_offspring = rank_sort(offspring_evaluation)

    xmean = recombination(mut_offspring[:,idx_rank_offspring[0: mu_rank_elite_size]])
    zmean = recombination(std_offspring[:,idx_rank_offspring[0: mu_rank_elite_size]])

    ps = (1 - cumulate_sigma_control) * ps + (math.sqrt(cumulate_sigma_control * (2 - cumulate_sigma_control) * mueff)) * (B.dot(zmean))
    hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cumulate_sigma_control) ** (2 * counteval / lambda_pop_size))/chiN < 1.4 + 2/(N + 1)
    pc = (1 - cov_adapt_time_constant) * pc + hsig * math.sqrt(cov_adapt_time_constant*(2-cov_adapt_time_constant) * mueff) * (B.dot(D.dot(zmean)))

    C = (1 - c1 - cmu) * C + c1
    C = C.dot(pc.dot(pc.T) + (1 - hsig) * cov_adapt_time_constant * (2 - cov_adapt_time_constant) * C)
    BDOffs = B.dot(D.dot(std_offspring[:, idx_rank_offspring[0: mu_rank_elite_size]]))
    C = C + cmu * (BDOffs.dot(np.diag(weights)).dot(BDOffs.T))

    sigma = sigma * np.exp((cumulate_sigma_control / damps) * (np.linalg.norm(ps) / chiN - 1))

    if (counteval - eigenval) > (lambda_pop_size / (1 + cmu) / N / 10):
      eigenval = counteval
      C = np.triu(C) + np.triu(C, 1).T
      [D, B] = np.linalg.eig(C)
      D = np.diag(np.sqrt(D))

    if offspring_evaluation[0] <= stopfitness:
      break

    if offspring_evaluation[0] == offspring_evaluation[math.floor(np.ceil(0.7 * lambda_pop_size))]:
      sigma = sigma * np.exp(0.2 * cumulate_sigma_control / damps)

  print("%s : %s" %(str(counteval), str(offspring_evaluation[0])))
  xmin = mut_offspring[:, idx_rank_offspring[0]]
  print(xmin)

if __name__ == "__main__":
  main()