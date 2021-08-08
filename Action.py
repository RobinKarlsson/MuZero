from numpy import zeros, bool

from typing import List
from Player import Player
from MuZeroConfig import MuZeroConfig

class Action(object):

  def __init__(self, index: int, player: Player, coordinates: List[int]):
    self.index = index
    self.player = player
    
    self.coordinates = coordinates

  def representation(self, gridsize: int):
    representation = zeros((2, gridsize, gridsize), dtype = bool)
    representation[0 if self.player.player == -1 else 1, self.coordinates] = 1
    return representation

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    return self.index == other.index

  def __gt__(self, other):
    return self.index > other.index


class ActionHistory(object):
  def __init__(self, history: List[Action] = [], action_space_size: int = 60):
    self.history = history
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action: Action):
    self.history.append(action)

  def last_action(self) -> Action:
    return self.history[-1]

  def to_play(self) -> Player:
    return Player()
