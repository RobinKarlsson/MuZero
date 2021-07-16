#python 3.8

class MinMax(object):
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    def newBoundary(self, x):
        self.minimum = x if x < self.minimum else self.minimum
        self.maximum = x if x > self.maximum else self.maximum

    def normalize(self, x):
        return (x - self.minimum) / (self.maximum - self.minimum) if self.maximum > self.minimum else x
        
