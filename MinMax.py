#python 3.8

from ctypes import CDLL, c_float
from os.path import abspath

so_file = "CLibrary/normalize.so"
f = CDLL(abspath(so_file))
f.normalize.argtypes = [c_float, c_float, c_float]
f.normalize.restype = c_float

class MinMax(object):
    def __init__(self, minimum: float, maximum: float):
        self.minimum = minimum
        self.maximum = maximum

    def newBoundary(self, x: float):
        self.minimum = x if x < self.minimum else self.minimum
        self.maximum = x if x > self.maximum else self.maximum

    def normalize(self, x: float) -> float:
        try:
            normalized = f.normalize(x, self.maximum, self.minimum)
        except Exception as inst:
            print('MinMax c normalize failed')
            print(type(inst))
            print(inst.args)
            print(inst)

            normalized = (x - self.minimum) / (self.maximum - self.minimum) if self.maximum > self.minimum else x
        return normalized

def _test():
    assert MinMax(0, 10).normalize(5) == 0.5

if __name__ == '__main__':
    _test()
