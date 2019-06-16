import scipy.special as spsp
import scipy.linalg as spla
import numpy as np


class CovarianceMatrixAdaptationEvolutionStrategy:

    def __init__(self, function, intialPoint,
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
                 generationsNumber=None):

        self.function = function
        self.intialPoint = intialPoint  # xmean
        self.dimensionsSize = intialPoint.size  # N

        if generationsNumber:  # stopeval
            self.generationsNumber = generationsNumber
        else:
            self.generationsNumber = 1000*(self.dimensionsSize**2)

        # sigma
        if stepSize:
            self.stepSize = stepSize
        else:
            self.stepSize = 0.3

        # lambda
        if populationSizeGeneration:
            self.populationSizeGeneration = populationSizeGeneration
        else:
            self.populationSizeGeneration = int(
                4 + np.floor(3 * np.log(self.dimensionsSize)))

        # mu
        if parentsNumber:
            self.parentsNumber = parentsNumber
        else:
            self.parentsNumber = self.populationSizeGeneration // 2

        # muXone array
        if recombinationWeights:
            self.recombinationWeights = recombinationWeights
        else:
            self.recombinationWeights = (np.log(0.5 * (self.populationSizeGeneration + 1)) -
                                         np.log(np.arange(1, self.populationSizeGeneration + 1)))

        normalizedWeights = (self.recombinationWeights[:self.parentsNumber] /
                             sum(self.recombinationWeights[:self.parentsNumber]))
        # mueff
        self.varianceEffectiveness = (
            sum(normalizedWeights)**2 / sum(normalizedWeights**2))

        # cs
        if learningRateCumulationStepSize:
            self.learningRateCumulationStepSize = learningRateCumulationStepSize
        else:
            self.learningRateCumulationStepSize = (self.varianceEffectiveness + 2) / \
                (self.dimensionsSize + self.varianceEffectiveness + 5)

        # damps
        if updateMitigationStepSize:
            self.updateMitigationStepSize = updateMitigationStepSize
        else:
            self.updateMitigationStepSize = (1 + 2 * max(0, np.sqrt((self.varianceEffectiveness - 1) /
                                                                    (self.dimensionsSize + 1)) - 1) +
                                             self.learningRateCumulationStepSize)

        # cc
        if learningRateCumulationRankOneUpdate:
            self.learningRateCumulationRankOneUpdate = learningRateCumulationRankOneUpdate
        else:
            self.learningRateCumulationRankOneUpdate = ((4 + self.varianceEffectiveness / self.dimensionsSize) /
                                                        (self.dimensionsSize + 4 + 2 * self.varianceEffectiveness / self.dimensionsSize))

        # c1
        if learningRateRankOneUpdate:
            self.learningRateRankOneUpdate = learningRateRankOneUpdate
        else:
            self.learningRateRankOneUpdate = 2 / ((self.dimensionsSize + 1.3) **
                                                  2 + self.varianceEffectiveness)

        # cmu
        if learningRateRankMuUpdate:
            self.learningRateRankMuUpdate = learningRateRankMuUpdate
        else:
            self.learningRateRankMuUpdate = min(1 - self.learningRateRankOneUpdate, 2 *
                                                ((self.varianceEffectiveness - 2 + 1 / self.varianceEffectiveness) /
                                                 ((self.dimensionsSize + 2)**2 + self.varianceEffectiveness)))

        if learningRateMeanVectorUpdate:
            self.learningRateMeanVectorUpdate = learningRateMeanVectorUpdate
        else:
            self.learningRateMeanVectorUpdate = 1

        if absoluteChangeEarlyStopping:
            self.absoluteChangeEarlyStopping = absoluteChangeEarlyStopping
        else:
            self.absoluteChangeEarlyStopping = 1e-5

        if relativeChangeEarlyStopping:
            self.relativeChangeEarlyStopping = relativeChangeEarlyStopping
        else:
            self.relativeChangeEarlyStopping = 1e-4

        if not recombinationWeights:
            negativeWeights = self.recombinationWeights[self.parentsNumber:]
            varianceEffectiveness_neg = (sum(negativeWeights)**2 /
                                         sum(negativeWeights**2))

            alphaMuNegative = 1 + self.learningRateRankOneUpdate / self.learningRateRankMuUpdate
            alphaVarianceEffectivenessNegative = 1 + 2 * \
                varianceEffectiveness_neg / (self.varianceEffectiveness + 2)

            alpha_pos_def_neg = ((1 - self.learningRateRankOneUpdate - self.learningRateRankMuUpdate) /
                                 (self.dimensionsSize * self.learningRateRankMuUpdate))

            self.recombinationWeights[:self.parentsNumber] = normalizedWeights

            denom = min(
                alphaMuNegative, alphaVarianceEffectivenessNegative, alpha_pos_def_neg)

            self.recombinationWeights[self.parentsNumber:] = (denom * negativeWeights /
                                                              -sum(negativeWeights))

        self.covarianceMatrix = np.identity(len(self.intialPoint))

        self.p_sig = 0
        self.p_c = 0

    def sample_and_evaluate(self, function, dimensionsSize, intialPoint, covarianceMatrix, populationSizeGeneration, stepSize):
        """ Samples the function evaluations

        Used to sample the candidate values and evaluates them.

        Args:
            function (numpy array -> float): Function to evaluate.
            dimensionsSize (int > 0): Number of dimensions.
            intialPoint (numpy array): Mean vector of the multivariate Gaussian
                distribution.
            covarianceMatrix (numpy array): Covariance matrix of the multivariate
                Gaussian distribution.
            populationSizeGeneration (int > 0): Number of samples.
            stepSize (float > 0): Overall variance step size.

        Returns:
            Numpy array containing the function values, the mean-centered and
            stepSize ajusted input values, and zero-centered and unadjusted
            input values as columns.
        """

        y = np.random.multivariate_normal(
            np.zeros(dimensionsSize), covarianceMatrix, size=populationSizeGeneration)

        x = intialPoint + stepSize * y
        func_values = list(map(function, x))
        conlearningRateMeanVectorUpdateatrix = np.c_[func_values, x, y]
        conlearningRateMeanVectorUpdateatrix = conlearningRateMeanVectorUpdateatrix[
            conlearningRateMeanVectorUpdateatrix[:, 0].argsort()]
        return conlearningRateMeanVectorUpdateatrix

    def heaviside(self, gen, norm_p_sig, learningRateCumulationStepSize, dimensionsSize, expec_norm_gaussian):
        """Heaviside function

        Prevents too fast increases of axes of the covariance matrix.

        Args:
            gen (int > 0): The current generation the minimization procedure is
                in.
            norm_p_sig(float > 0): The norm of p_sig.
            learningRateCumulationStepSize (float > 0): Learning rate for the cumulation for the
                step-size control.
            dimensionsSize (int > 0): Number of dimensions.
            expec_norm_gaussian (float > 0): Expectation of the norm of a
                multivariate standard normal distribution.

        Returns:
            0 or 1.

        """
        a = norm_p_sig / \
            (np.sqrt(1 - (1 - learningRateCumulationStepSize)**(2 * (gen + 1))))

        b = (1.4 + 2 / (dimensionsSize + 1)) * expec_norm_gaussian

        if a < b:
            return 1
        else:
            return 0

    def minimizar(self):

        # Matrix to fix some numerical issues with the cholesky decomposition.
        offset_matrix = np.identity(self.dimensionsSize) * 0.1
        expec_norm_gaussian = (np.sqrt(2) * spsp.gamma(0.5 * (self.dimensionsSize + 1)) /
                               spsp.gamma(0.5 * self.dimensionsSize))
        pop_matrix = self.sample_and_evaluate(self.function, self.dimensionsSize,
                                              self.intialPoint, self.covarianceMatrix,
                                              self.populationSizeGeneration, self.stepSize)

        y = pop_matrix[:, (self.dimensionsSize + 1):]
        x = pop_matrix[:, 1:(self.dimensionsSize + 1)]
        func_values = pop_matrix[:, 0]
        best_fvalue = pop_matrix[0, 0]
        best_param = x[0, :]

        for gen in range(1, self.generationsNumber + 1):

            y_weighted = np.sum(
                y[:self.parentsNumber] * self.recombinationWeights[:self.parentsNumber, np.newaxis], axis=0
            )

            upd_intialPoint = (
                self.intialPoint + self.learningRateMeanVectorUpdate * self.stepSize * y_weighted
            )

            diff_vec = abs(upd_intialPoint - self.intialPoint)

            if (max(diff_vec) < self.absoluteChangeEarlyStopping or max(diff_vec / abs(self.intialPoint)) < self.relativeChangeEarlyStopping):

                result_dict = {'best fvalue': best_fvalue,
                               'best param': best_param,
                               'mean vector': self.intialPoint,
                               'cov matrix': self.covarianceMatrix,
                               'converged in': gen,
                               'abs/rel conv': (max(diff_vec),
                                                max(diff_vec /
                                                    abs(self.intialPoint)))}
                return result_dict

            self.intialPoint = upd_intialPoint

            try:
                chol_covm_L = spla.cholesky(self.covarianceMatrix, lower=True)
            except np.linalg.linalg.LinAlgError as err:

                print('Exceção na decomposição de Cholesky', err)
                try:
                    chol_covm_L = spla.cholesky(
                        self.covarianceMatrix + offset_matrix, lower=True)
                except BaseException:
                    print('Matriz com a diagonal aumentada também falhou.')

            c_y_vec = spla.solve_triangular(
                chol_covm_L, y_weighted, lower=True
            )

            # Updating the step sizes
            self.p_sig = ((1 - self.learningRateRankMuUpdate) * self.p_sig +
                          np.sqrt(self.learningRateCumulationStepSize * (2 - self.learningRateCumulationStepSize) *
                                  self.varianceEffectiveness) * c_y_vec)
            norm_p_sig = np.linalg.norm(self.p_sig)
            self.stepSize = (self.stepSize *
                             np.exp(self.learningRateCumulationStepSize / self.updateMitigationStepSize *
                                    norm_p_sig /
                                    expec_norm_gaussian - 1))

            # Updating the covariance matrix
            h_sig = self.heaviside(gen, norm_p_sig, self.learningRateCumulationStepSize,
                                   self.dimensionsSize, expec_norm_gaussian)

            self.p_c = ((1 - self.learningRateCumulationRankOneUpdate) * self.p_c +
                        h_sig * np.sqrt(self.learningRateCumulationRankOneUpdate * (2 - self.learningRateCumulationRankOneUpdate) *
                                        self.varianceEffectiveness) * y_weighted)
            c_y_mat = spla.solve_triangular(chol_covm_L,
                                            y[self.parentsNumber:, :].T,
                                            lower=True).T
            c_y_mat_norm = np.linalg.norm(c_y_mat, axis=1)

            w_adj = self.recombinationWeights.copy()
            w_adj[self.parentsNumber:] = (w_adj[self.parentsNumber:] *
                                          (self.dimensionsSize / c_y_mat_norm))

            matrix_sum = (w_adj * y.T).dot(y)
            self.covarianceMatrix = ((1 + self.learningRateRankOneUpdate * (1 - h_sig) * self.learningRateCumulationRankOneUpdate *
                                      (2 - self.learningRateCumulationRankOneUpdate) - self.learningRateRankOneUpdate - self.learningRateRankMuUpdate *
                                      sum(self.recombinationWeights)) * self.covarianceMatrix + self.learningRateRankOneUpdate *
                                     np.outer(self.p_c, self.p_c)
                                     + self.learningRateRankMuUpdate * matrix_sum)

            # Create new generations
            pop_matrix = self.sample_and_evaluate(self.function,
                                                  self.dimensionsSize,
                                                  self.intialPoint,
                                                  self.covarianceMatrix,
                                                  self.populationSizeGeneration,
                                                  self.stepSize)

            y = pop_matrix[:, (self.dimensionsSize + 1):]
            x = pop_matrix[:, 1:(self.dimensionsSize + 1)]
            func_values = pop_matrix[:, 0]

            if func_values[0] < best_fvalue:
                best_fvalue = pop_matrix[0, 0]
                best_param = x[0, :]

            print('Melhor valor:', best_fvalue, ' Geração:', gen)

        else:
            print('Não houve convergência para o número de gerações escolhido.')
            result_dict = {'Melhor valor da função': best_fvalue,
                           'Melhor parametro': best_param,
                           'Vetor de ponto inicial': self.intialPoint,
                           'Matriz de covariancia': self.covarianceMatrix,
                           'Número de iterações': self.generationsNumber}
            return result_dict
