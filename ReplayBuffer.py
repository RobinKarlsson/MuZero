#python 3.8

from numpy.random import choice

from MuZeroConfig import MuZeroConfig
from MuZeroGameWrapper import MuZeroGameWrapper

class ReplayBuffer(object):

    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []
        self.config = config

    def save_game(self, wrapper: MuZeroGameWrapper):
        if len(self.buffer) > self.window_size:
          self.buffer.pop(0)
        self.buffer.append(wrapper)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        if(self.batch_size > len(self.buffer)):
            batch_size = len(self.buffer)
        else:
            batch_size = self.batch_size
            
        games = choice(self.buffer, batch_size, replace=False)
        game_pos = [(g, self.sample_position(g)) for g in games]
        
        return [(g.getImage(i),
                 g.action_history.history[i:i + num_unroll_steps],
                 g.getTargets(i, num_unroll_steps, td_steps, g.currentPlayer())) for (g, i) in game_pos]

    def sample_game(self):
        # Random sample game from buffer
        return choice(self.buffer)

    def sample_position(self, game) -> int:
        # Random sample position from game
        return choice(len(game.state_history) - self.config.num_unroll_steps)
