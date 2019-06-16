import numpy as np
import math


class Run:
    def __init__(self, model, func, stopfitness=1e-10, optim='min'):
        self.model = model
        self.fn_eval = func
        self.stopfitness = stopfitness
        self.optim = optim

    def run(self, fold):
        idx_rank_offspring = None
        model = self.model
        counteval = 0
        eigenval = 0

        while(not self.__iteration_stop_condition__(counteval, model.stopeval)):
            [std_offspring, mut_offspring, idx_rank_offspring, fitness, counteval,
                eigenval] = model.next(self.fn_eval, counteval, eigenval, fold, self.optim)
            xmin = mut_offspring[:, idx_rank_offspring[0]]
            real_fitness = 1.0 / fitness
            print("%s : %s" % (str(counteval), str(real_fitness)))

            if self.__fitness_stop_condition__(real_fitness):
                break

        return [xmin, real_fitness, counteval]

    def __iteration_stop_condition__(self, counteval, stopeval):
        return counteval >= stopeval

    def __fitness_stop_condition__(self, fitness):
        return (fitness <= self.stopfitness) if self.optim is 'min' else (fitness > self.stopfitness)
