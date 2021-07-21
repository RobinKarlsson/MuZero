
class Player(object):
    def __init__(self, player: int):
        self.player = player

    def __repr__(self) -> int:
        return self.player

    def __eq__(self, other):
        return self.player == other.player

def _test():
    p1 = Player(1)
    p2 = Player(2)

    assert p1 != p2
    assert p1 == Player(1)

if __name__ == '__main__':
    _test()
