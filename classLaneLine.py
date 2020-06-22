from collections import deque

def create_queue(length = 10):
    return deque(maxlen=length)

class LaneLine:
    def __init__(self):
        
        self.polynomial_coeff = None
        self.line_fit_x = None
        self.non_zero_x = []
        self.non_zero_y = []
        self.windows = []    