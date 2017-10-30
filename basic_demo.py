from sweep import Sweeper

class Model():
    def __init__(self, params):
        self.params = params
        self.targets = {
            'a': 4,
            'b': 3.5,
            'c': 10**-1.2,
            'd': 10**-3.7,
            'e': 10**-3.7,
            'f': 5,
        }
    def train(self):
        pass

    def eval(self):
        return sum([abs(self.params[k] / self.targets[k] - 1) for k in self.params.keys()])

if __name__ == '__main__':
    params = {
                'a': {'type': 'discrete', 'range': [4,5,6]},
                'b': {'type': 'continuous', 'range': [1, 5], 'sweep_type': 'linear', 'sweep_num': 5},
                'c': {'type': 'continuous', 'range': [1e-2, 1], 'sweep_type': 'exp', 'sweep_num': 5},
                'd': {'type': 'continuous', 'range': [1e-4, 1e-2], 'sweep_type': 'exp', 'sweep_num': 5},
                'e': {'type': 'continuous', 'range': [1e-4, 1e-2], 'sweep_type': 'log', 'sweep_num': 5},
                'f': {'type': 'discrete', 'range': [4,5,6]},
                }
    s = Sweeper(params, Model)
    print('\n\n\nbest, random:', s.random_sweep(num_iters=10))
    print('target was:', sorted(Model(None).targets.items()))
    input('Press Enter to continue. ')
    print('\n\n\nbest, grid:', s.grid_sweep(num_iters=10))
    print('target was:', sorted(Model(None).targets.items()))

