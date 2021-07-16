
class Player(object):
    def __init__(self, player: int):
        self.player = player

    def __repr__(self) -> int:
        return self.player

    def __eq__(self, other):
        return self.player == other.player
