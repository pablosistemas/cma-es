import numpy as np
import re

class SKLearnEvaluation():
    def __init__(self, string_parameter, load_pipeline, evaluate_pipeline):
        self.load_pipeline = load_pipeline
        self.evaluate_pipeline = evaluate_pipeline
        self.array_parameter = string_parameter.split()

    def run(self, x, fold):
        candidate = self.load_pipeline(self.get_pipeline_from(x))
        candidate_p = self.parse_integers_in_evaluation(candidate)
        results = self.evaluate_pipeline(self.parse_integers_in_evaluation(candidate_p), fold)

        if results == 'invalid':
            norm_value = 0
            for x_v in x:
                if x_v < 0:
                    norm_value = norm_value + np.abs(x_v)
            base_bad_value = 10
            repair_alpha = self.__get_fraction_of_value__(base_bad_value, np.abs(int(x_v)))
            return base_bad_value + repair_alpha * (norm_value ** 2)

        return 1.0 / results['f1_weighted']['mean']

    def get_pipeline_from(self, x):
        idx_x = 0
        for idx in range(len(self.array_parameter)):
            try:
                # if current idx parameter is float, then substitute it
                float(self.array_parameter[idx])
                float_of_x = float(x[idx_x])
                self.array_parameter[idx] = str(float_of_x)
                idx_x = idx_x + 1
            except:
                pass

        return ' '.join(self.array_parameter)

    def parse_integers_in_evaluation(self, string_eval):
        string_eval_p = string_eval
        for pattern in ['n_components', 'iterated_power', 'k', 'max_iter', 'penalty', 'max_depth', 'max_leaf_nodes', 'degree']:
            base_string_match_value_rgx = '.*"%s":\s*(\d[.\d]+).*'%(pattern)
            base_string_match_all_rgx = '(.*)("%s":\s*\d[.\d]+)(.*)'%(pattern)
            obj_value = re.match(base_string_match_value_rgx, string_eval_p)
            obj_string = re.match(base_string_match_all_rgx, string_eval_p)
            if obj_value is not None:
                string_eval_p = re.sub(base_string_match_all_rgx, '%s%s%s%s'%(obj_string.group(1), '"%s":'%(pattern), int(float(obj_value.group(1))), obj_string.group(3)), string_eval)
        return string_eval_p
    
    def __get_fraction_of_value__(self, v, fract_count):
        return v >> fract_count
