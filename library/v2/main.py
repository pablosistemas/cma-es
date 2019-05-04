import numpy as np
import random
from cmaes import CovarianceMatrixAdaptationEvolutionStrategy


def initialSerachPointGenerator(dimensions, startValue=0, endValue=1):
    randomArray = randomArrayGenerator(dimensions, startValue, endValue)
    return np.array(randomArray)


def randomArrayGenerator(dimensions, startValue=0, endValue=1):
    valuesArray = []
    while(dimensions > 0):
        valuesArray.append(random.uniform(startValue, endValue))
        dimensions -= 1
    return valuesArray


def objectiveFunction(vector):
    param1, param2 = vector
    return param1**2 + param2**2


dimensionsNumber = 2
initialMeanVector = initialSerachPointGenerator(dimensionsNumber)

cma_es = CovarianceMatrixAdaptationEvolutionStrategy(function=objectiveFunction,
                                                     intialPoint=initialMeanVector,
                                                     stepSize=None,
                                                     populationSizeGeneration=None,
                                                     parentsNumber=None,
                                                     recombinationWeights=None,
                                                     learningRateCumulationStepSize=None,
                                                     updateMitigationStepSize=None,
                                                     learningRateCumulationRankOneUpdate=None,
                                                     learningRateRankOneUpdate=None,
                                                     learningRateRankMuUpdate=None,
                                                     learningRateMeanVectorUpdate=None,
                                                     absoluteChangeEarlyStopping=None,
                                                     relativeChangeEarlyStopping=None,
                                                     generationsNumber=None)

result = cma_es.minimizar()

print('\n', result)
