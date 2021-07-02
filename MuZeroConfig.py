#python 3.8

def visit_softmax_temperature(moves):
    return 1. if moves < 6 else 0

class MuZeroConfig(object):

    def __init__(self,
                 action_space_size: int = 60,
                 max_moves: int = 60,
                 board_size: int = 64,
                 discount: float = 1.,
                 dirichlet_alpha: float = 3e-2,
                 num_simulations: int = 1,
                 batch_size: int = 512,
                 td_steps: int = 60,
                 lr_init: float = 1e-2,
                 lr_decay_steps: float = 4e5,
                 visit_softmax_temperature_fn = visit_softmax_temperature,
                 window_size: int = 512,
                 training_steps: int = 1e5,
                 checkpoint_interval: int = 1e3,
                 pb_c_base: float = 19652,
                 pb_c_init: float = 1.25,
                 root_exploration_fraction: float = 0.25,
                 num_unroll_steps: int = 5,
                 weight_decay: float = 1e-4,
                 momentum: float = 0.9,
                 lr_decay_rate: float = 0.1,
                 channels: int = 128):
        
        ### Self-Play
        self.action_space_size = action_space_size

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction

        # UCB formula
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init

        ### Training
        self.training_steps = training_steps
        self.checkpoint_interval = checkpoint_interval
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps
        self.channels = channels

        self.weight_decay = weight_decay
        self.momentum = momentum

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps

