#python 3.8

class MuZeroConfig(object):

    def __init__(self,
                 max_moves: int = 8**2-4, #max number of moves possible
                 board_gridsize: int = 8, #number of rows/columns on board
                 action_space_size: int = 8**2+1, #number of possible actions
                 discount: float = 0.99, #discount on node values while propagating up mcts tree
                 dirichlet_alpha: float = 0.3, #exploration noice on root node actions to have mcts explore new options
                 consider_backward_states: int = 4, #backward states in image of board
                 num_simulations: int = 8**2, #number of mcts iterations
                 num_threads: int = 1, #number of selfplay threads
                 num_selfplay: int = None, 
                 refresh_replaybuffer: int = 5,
                 batch_size: int = 8**3, #number of played games too store in replaybuffer
                 td_steps: int = 8*8-4, #number of future moves for bootstraping
                 lr_init: float = 1e-2, #initial learning rate
                 lr_decay_rate: float = 1e-1, #learning rate decay rate
                 lr_decay_steps: float = 4e5, #learning rate decay steps
                 weight_decay: float = 1e-4,
                 momentum: float = 0.9, #momentum for sgd
                 softmax_threshold = 21, #threshold between probability of selection for each action by number of visits and max visits after mcts search
                 window_size: int = 8**3, #max number of games stored in replaybuffer
                 training_steps: int = 1e6, #number of epochs to train network
                 checkpoint_interval: int = 1e3, #save network interval while training
                 root_exploration_fraction: float = 0.25,
                 num_unroll_steps: int = 4,
                 pb_c_base: float = 19652,
                 pb_c_init: float = 1.25,
                 channels: int = 128,
                 boundary_min: int = -1,
                 boundary_max: int = 1,
                 torch_device: str = 'cpu'):
        
        ### Self-Play
        self.num_threads = num_threads
        self.num_selfplay = num_selfplay if num_selfplay else int(window_size / 5)
        self.refresh_replaybuffer = refresh_replaybuffer

        self.board_gridsize = board_gridsize
        self.action_space_size = action_space_size
        self.max_moves = max_moves
        self.board_size = board_gridsize * board_gridsize
        self.num_simulations = num_simulations
        self.discount = discount
        self.consider_backward_states = consider_backward_states

        self.boundary_min = boundary_min
        self.boundary_max = boundary_max

        self.softmax_threshold = softmax_threshold

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

