from library.model.cmaes import Cmaes as Cmaes
from library.algorithm.run import Run as Run
from library.evaluation.felli import Felli as Felli
from library.evaluation.cigar import Cigar as Cigar
from library.evolution.generation import NextCmaes as Generation
from library.evolution.recombination import Recombination as Recombination

import numpy as np
import math

def main():

  dimension_size = 10
  model = Cmaes(dimension_size, Recombination().run)
  algorithm = Run(model, Cigar().run)
  [xmin, fitness, counteval] = algorithm.run()

  print("%s : %s" %(str(counteval), str(fitness)))
  print(xmin)

  return

if __name__ == "__main__":
  main()