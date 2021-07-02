from typing import List
from Player import Player

class Action(object):

  def __init__(self, index: int, coordinates: List[int], player: Player):
    self.index = index
    self.player = player
    self.coordinates = coordinates

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    return self.index == other.index

  def __gt__(self, other):
    return self.index > other.index


class ActionHistory(object):
  def __init__(self, history: List[Action] = [], action_space_size: int = 0):
    self.history = history
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action: Action):
    self.history.append(action)

  def last_action(self) -> Action:
    return self.history[-1]

  def action_space(self) -> List[Action]:
    return [Action(i) for i in range(self.action_space_size)]

  def to_play(self) -> Player:
    return Player()
