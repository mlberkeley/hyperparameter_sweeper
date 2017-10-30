## Automated Hyperparameter Sweeper   
### Usage:   
- Create a Sweeper() instance (see sweep.py for documentation on parameters).
- Create a wrapper Model() class which is instantiated with a param_dict, can be trained with Model.train(), and outputs a loss on Model.eval().
- Call the sweeper instance with random_sweep() or grid_sweep().
- See basic_demo.py for example usage with a dummy Model() class.
- See mnist_demo.py for example usage with MNIST.
   - Requires [pytorch](http://pytorch.org/)).
   - Requires GPU + CUDA.
