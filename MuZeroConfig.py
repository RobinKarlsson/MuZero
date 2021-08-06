#python 3.8

def visit_softmax_temperature(moves):
    return 1. if moves < 6 else 0

class MuZeroConfig(object):

    def __init__(self,
                 max_moves: int = 8**2-4,
                 board_gridsize: int = 8,
                 action_space_size: int = 8**2,
                 discount: float = 0.99,
                 dirichlet_alpha: float = 0.3,
                 consider_backward_states: int = 4,
                 num_simulations: int = 200,
                 num_threads: int = 1,
                 num_selfplay: int = None,
                 refresh_replaybuffer: int = 5,
                 batch_size: int = 500,
                 td_steps: int = 8*8-4,
                 lr_init: float = 5e-3,
                 lr_decay_rate: float = 1,
                 lr_decay_steps: float = 1e4,
                 visit_softmax_temperature_fn = visit_softmax_temperature,
                 window_size: int = 500,
                 training_steps: int = 1e6,
                 checkpoint_interval: int = 1e2,
                 pb_c_base: float = 19652,
                 pb_c_init: float = 1.25,
                 root_exploration_fraction: float = 0.25,
                 num_unroll_steps: int = 5,
                 weight_decay: float = 1e-4,
                 momentum: float = 0.9,
                 channels: int = 128,
                 boundary_min: int = -1,
                 boundary_max: int = 1,
                 torch_device: str = 'cpu'):
        
        ### Self-Play
        self.num_threads = num_threads
        self.num_selfplay = num_selfplay if num_selfplay else int(window_size / 5)
        self.refresh_replaybuffer = refresh_replaybuffer
        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn

        self.board_gridsize = board_gridsize
        self.action_space_size = action_space_size
        self.max_moves = max_moves
        self.board_size = board_gridsize * board_gridsize
        self.num_simulations = num_simulations
        self.discount = discount
        self.consider_backward_states = consider_backward_states

        self.boundary_min = boundary_min
        self.boundary_max = boundary_max

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction

        # UCB formula
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init

        ### Training
        self.training_steps = int(training_steps)
        self.checkpoint_interval = int(checkpoint_interval)
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps
        self.channels = channels

        self.weight_decay = weight_decay
        self.momentum = momentum

        self.torch_device = torch_device

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps

