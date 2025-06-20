# Configurations for different experiments

# Configuration for Regression
class RegConfig:
    train_size = 1024                    # num of training samples
    batch_size = 128            
    lr = 1e-3  
    epochs = 50
    train_samples = 10                   # num of train samples
    test_samples = 10                    # num of test samples
    num_test_points = 400                # num of test points
    experiment = 'regression'            # experiment type
    hidden_units = 400                   # num of hidden units
    noise_tolerance = 1.0                # log likelihood sigma
    mu_init = [-0.2, 0.2]                # mean initialization range 
    rho_init = [-4, -3]                  # rho_param initialization range

    # In the paper the initialization is for -log(stddev_1) and -log(stddev_2). Here we pass the negative values directly.
    prior_init = [0.5, -0, -6]           # mixture weight, log(stddev_1), log(stddev_2)

# Configuration for Reinforcement Learning
class RLConfig:
    data_dir = './dataset/agaricus-lepiota.data'
    batch_size = 64
    num_batches = 64
    buffer_size = batch_size * num_batches  # buffer for latest batch of mushrooms
    lr = 1e-4
    training_steps = 50000
    experiment = 'regression'               # experiment type
    hidden_units = 100                      # num of hidden units
    mu_init = [-0.2, 0.2]                   # mean initialization range
    rho_init = [-5, -4]                     # rho_param initialization range
    prior_init = [0.5, -0, -6]              # mixture weight, log(stddev_1), log(stddev_2)

# Configuration for Classification
class ClassConfig:
    batch_size = 128
    lr = 1e-3 
    epochs = 600 
    hidden_units = 1200
    experiment = 'classification'           # experiment type
    dropout = False
    train_samples = 1 
    test_samples = 10
    x_shape = 28 * 28                       # x shape
    classes = 10                            # number of output classes
    mu_init = [-0.2, 0.2]                   # mean initialization range
    rho_init = [-5, -4]                     # rho_param initialization range
    prior_init = [0.75, 0, -7]              # mixture weight, log(stddev_1), log(stddev_2)