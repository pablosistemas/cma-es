import numpy as np

class GetNumericParameters:
    @staticmethod
    def get(string_with_candidate_solution):
        array_with_candidate_solution = string_with_candidate_solution.split()
        return GetNumericParameters().get_numeric_only_from_array(array_with_candidate_solution)
    
    @staticmethod
    def get_numeric_only_from_array(array_with_candidate_solution):
        array_numeric_only = []
        for parameter in array_with_candidate_solution:
            try:
                array_numeric_only.append(float(parameter))
            except:
                print('Non numeric parameter')

        return np.array(array_numeric_only)