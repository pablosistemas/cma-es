import numpy as np
import math

class Run:
  def __init__(self, model, func):
    self.model = model
    self.fn_eval = func
    self.stopfitness = 1e-10

  def run(self):
    idx_rank_offspring = None
    model = self.model
    counteval = 0
    eigenval = 0

    while(not self.__stop_condition(counteval, model.stopeval)):
      [std_offspring, mut_offspring, idx_rank_offspring, fitness, counteval, eigenval] = model.next(self.fn_eval, counteval, eigenval)
      xmin = mut_offspring[:, idx_rank_offspring[0]]
      print("%s : %s" %(str(counteval), str(fitness)))
      if fitness <= self.stopfitness:
        break


    return [xmin, fitness, counteval]

  def __stop_condition(self, counteval, stopeval):
    return counteval >= stopeval
