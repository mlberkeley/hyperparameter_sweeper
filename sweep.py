import numpy as np
import copy
import os

DEFAULTS = {
    'SWEEP_TYPE': 'exp',
    'SWEEP_NUM': 5,
    'SQUEEZE_FACTOR': 0.8,
}

class Sweeper(object):
    """ param_dict should contain entries of the form:
        {param_name: {type, range, sweep_type, sweep_num}},
        WHERE:
        param_name is:
            a string,
        type is:
            'discrete' or 'continuous',
        range is:
            EITHER
                a list of possible values (if type == 'discrete')
            OR
                the start and end of the possible range (if type == 'continuous')
                    (must have start > 0 and start < end)
        sweep_type (for type == 'continuous' only) is:
            'linear'
            'exp' (e.g. 1e-8, 1e-7, 1e-6, ...)
            'log' (e.g. log(1e-2), log(2e-2), log(3e-2), ...)
        sweep_num (for type == 'continuous' only) is:
            an integer
                - in grid sweep, we check sweep_num values in the possible range,
                    distributed according to sweep_type)
                - in random sweep, we use this to calculate how many values to
                    sample
         squeeze_factor (for type=='continuous' only) is:
            a float
                - how much to shrink the search range after each iteration


        model_create_fn should create a Model with desired parameters,
        WHERE:
            we can call Model.train() and Model.eval() on a predefined dataset
                - Model.eval() should return the loss

        top_n is:
            a float
                - the fraction of the top scores from the current iteration
                    that are used to determine the range for the next iteration

        log_file is:
            a string
                - path to file to write log to
    """
    def __init__(self, param_dict, model_create_fn,  log_file='sweep_log.txt', top_n=0.2):
        self.param_dict = param_dict
        for param, vals in filter(lambda kv: kv[1]['type'] == 'continuous', self.param_dict.items()):
            if 'squeeze_factor' not in vals:
                self.param_dict[param]['squeeze_factor'] = DEFAULTS['SQUEEZE_FACTOR']
            if 'sweep_type' not in vals:
                self.param_dict[param]['sweep_type'] = DEFAULTS['SWEEP_TYPE']
            if 'sweep_num' not in vals:
                self.param_dict[param]['sweep_num'] = DEFAULTS['SWEEP_NUM']
                
        self.model_create_fn = model_create_fn
        self.top_n = top_n
        while os.path.isfile(log_file):
            overwrite = input("Log file '{}' already exists. Overwrite? [y/N] ".format(log_file))
            if overwrite in ["y", "Y"]:
                os.remove(log_file)
                print("Log file overwritten.")
            else:
                new_log_file = input("Enter new log file name [{}]: ".format(log_file))
                if new_log_file:
                    log_file = new_log_file
        self.log_file = log_file

    def eval(self, params):
        return self.eval(params)

    def grid_sweep(self, num_iters=5):
        continuous_params = filter(lambda kv: kv[1]['type'] == 'continuous', self.param_dict.items())
        continuous_ranges = {kv[0]: kv[1]['range'] for kv in continuous_params}
        return self.sweep(num_iters=num_iters, method='grid', results={}, continuous_ranges=continuous_ranges)

    def random_sweep(self, num_iters=5):
        continuous_params = filter(lambda kv: kv[1]['type'] == 'continuous', self.param_dict.items())
        continuous_ranges = {kv[0]: kv[1]['range'] for kv in continuous_params}
        return self.sweep(num_iters=num_iters, method='random', results={}, continuous_ranges=continuous_ranges)

    def sweep(self, continuous_ranges, num_iters=1, method='random', results={}):
        if not num_iters:
            res = min(results.items(), key=lambda kv: kv[1])
            return sorted(res[0]), res[1]

        # get combinations of values to sweep through for this iteration
        if continuous_ranges:
            discrete_combs = self._get_discrete_combs()
            if method == 'random':
                continuous_combs = self._sample_continuous(continuous_ranges)
            elif method == 'grid':
                continuous_combs = self._discretize_continuous(continuous_ranges)
            combs = []
            for d_comb in discrete_combs:
                for c_comb in continuous_combs:
                    comb = copy.deepcopy(d_comb)
                    for param, val in c_comb.items():
                        comb[param] = val
                    combs.append(comb)
        else:
            combs = self._get_discrete_combs()
        print('\n\n\nStarting sweep, {} combinations'.format(len(combs)))
        print('Continuous ranges for this iteration are: \n{}'.format(continuous_ranges))

        # collect and log results
        thisResults = {}
        for comb in combs:
            model = self.model_create_fn(comb)
            model.train()
            results[tuple(sorted(comb.items()))] = model.eval()
            thisResults[tuple(sorted(comb.items()))] = results[tuple(sorted(comb.items()))]
        with open(self.log_file, 'a') as f:
            for comb, result in thisResults.items():
                f.write(str(comb) + ': ' + str(result) + '\n')
            f.write('=' * 80 + '\n')

        # get top_n results, and narrow the range
        sortedResults = sorted(thisResults.keys(), key=lambda k: thisResults[k])
        sortedResults = sortedResults[:int(len(sortedResults) * self.top_n) + 1]
        combs = [{pv[0]: pv[1] for pv in comb} for comb in sortedResults]
        for param in list(continuous_ranges.keys()):
            if self.param_dict[param]['sweep_type'] == 'linear':
                #var = np.var([comb[param+'discrete'] for comb in combs])
                mean = np.mean([comb[param] for comb in combs])
                span = continuous_ranges[param][1] - continuous_ranges[param][0]
                continuous_ranges[param] = [
                    max(1e-12, mean - span/2.*self.param_dict[param]['squeeze_factor']),
                    mean + span/2.*self.param_dict[param]['squeeze_factor']
                ]
            elif self.param_dict[param]['sweep_type'] == 'exp':
                #var = np.var([np.log(comb[param+'discrete']) for comb in combs])
                mean = np.mean([np.log(comb[param]) for comb in combs])
                span = np.log(continuous_ranges[param][1]) - np.log(continuous_ranges[param][0])
                continuous_ranges[param] = [
                    max(1e-12, np.exp(mean - span/2.*self.param_dict[param]['squeeze_factor'])),
                    np.exp(mean + span/2.*self.param_dict[param]['squeeze_factor'])
                ]
            elif self.param_dict[param]['sweep_type'] == 'log':
                #var = np.var([np.exp(comb[param+'discrete']) for comb in combs])
                mean = np.mean([np.exp(comb[param]) for comb in combs])
                span = np.exp(continuous_ranges[param][1]) - np.exp(continuous_ranges[param][0])
                continuous_ranges[param] = [
                    max(1e-12, np.log(mean - span/2.*self.param_dict[param]['squeeze_factor'])),
                    np.log(mean + span/2.*self.param_dict[param]['squeeze_factor'])
                ]
        return self.sweep(continuous_ranges=continuous_ranges, num_iters=num_iters-1, method=method, results=results)

    def _get_discrete_combs(self):
        combs = [{}]
        for param, vals in filter(lambda kv: kv[1]['type'] == 'discrete', self.param_dict.items()):
            param_range = vals['range']
            oldcombs = copy.deepcopy(combs)
            combs = []
            for oldcomb in oldcombs:
                for val in param_range:
                    newcomb = copy.deepcopy(oldcomb)
                    newcomb[param] = val
                    combs.append(newcomb)
        return combs

    def _discretize_continuous(self, continuous_ranges):
        combs = [{}]
        for param, param_range in continuous_ranges.items():
            sweep_type = self.param_dict[param]['sweep_type']
            sweep_num = self.param_dict[param]['sweep_num']
            if sweep_type == 'linear':
                param_range = np.linspace(param_range[0], param_range[1], num=sweep_num, endpoint=True).tolist()
            elif sweep_type == 'exp':
                param_range = np.exp(np.linspace(np.log(param_range[0]), np.log(param_range[1]), num=sweep_num, endpoint=True)).tolist()
            elif sweep_type == 'log':
                param_range = np.log(np.linspace(np.exp(param_range[0]), np.exp(param_range[1]), num=sweep_num, endpoint=True)).tolist()

            oldcombs = copy.deepcopy(combs)
            combs = []
            for oldcomb in oldcombs:
                for val in param_range:
                    newcomb = copy.deepcopy(oldcomb)
                    newcomb[param] = val
                    combs.append(newcomb)
        return combs

    def _sample_continuous(self, continuous_ranges):
        samples = {}
        num_samples = 1
        for param, vals in filter(lambda kv: kv[1]['type'] == 'continuous', self.param_dict.items()):
            sweep_num = self.param_dict[param]['sweep_num']
            num_samples *= sweep_num

        for param, param_range in continuous_ranges.items():
            sweep_type = self.param_dict[param]['sweep_type']
            if sweep_type == 'linear':
                span = param_range[1] - param_range[0]
                k = span / float(sweep_num - 1)
                samples[param] = np.random.uniform(param_range[0]-k, param_range[1]+k, size=num_samples).tolist()
            elif sweep_type == 'exp':
                span = np.log(param_range[1]) - np.log(param_range[0])
                k = span / float(sweep_num - 1)
                samples[param] = np.exp(np.random.uniform(np.log(param_range[0])-k, np.log(param_range[1])+k, size=num_samples)).tolist()
            elif sweep_type == 'log':
                span = np.exp(param_range[1]) - np.exp(param_range[0])
                k = span / float(sweep_num - 1)
                samples[param] = np.log(np.random.uniform(np.exp(param_range[0])-k, np.exp(param_range[1])+k, size=num_samples)).tolist()

        res = []
        for i in range(num_samples):
            res.append({param: vals[i] for param, vals in samples.items()})
        return res
