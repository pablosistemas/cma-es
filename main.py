from library.model.cmaes import Cmaes as Cmaes
from library.algorithm.run import Run as Run
from library.evaluation.felli import Felli as Felli
from library.evolution.generation import Next as Generation
from library.evolution.recombination import Recombination as Recombination

import numpy as np
import math

def main():

  dimension_size = 10
  model = Cmaes(dimension_size)
  evaluation_fn = Felli()
  algorithm = Run(model, evaluation_fn.run, Generation.run, Recombination.run)
  [xmin, counteval] = algorithm.run()

  # print("%s : %s" %(str(counteval), str(offspring_evaluation[0])))
  print(xmin)

  return

if __name__ == "__main__":
  main()