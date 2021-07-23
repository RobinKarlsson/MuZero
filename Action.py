from numpy import zeros, bool
#from copy import deepcopy
from pickle import loads, dumps

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
    #return ActionHistory(deepcopy(self.history), self.action_space_size)
    #return ActionHistory(loads(dumps(self.history)), self.action_space_size)
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action: Action):
    self.history.append(action)

  def last_action(self) -> Action:
    return self.history[-1]

  def to_play(self) -> Player:
    return Player()

def _test():
  a1 = Action(1, 1, [1,1])
  a2 = Action(2, 2, [2,2])
  assert a1 != a2
  a3 = Action(3, 3, [3,3])
  h = ActionHistory([a1])

  a_list = [a1,a2]
  assert a3 not in a_list
  assert a1 in a_list

  assert len(h.history) == 1
  h.add_action(a1)
  assert len(h.history) == 2

  h_new = h.clone()
  assert len(h_new.history) == 2
  h_new.add_action(a3)
  assert len(h_new.history) == len(h.history) + 1
  assert h.history[0] == h_new.history[0]

if __name__ == '__main__':
  _test()
