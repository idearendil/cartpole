from collections import deque
import random


class Replay_Buffer():
    """
    Class of replay buffer.
    This replay buffer includes functions such as push and pull.
    Each data consists of s, a, r, s', o.
    (if o is 0, then s' is terminal state.)
    """
    def __init__(self, device, max_size):
        self.buffer = deque(maxlen=max_size)
        self.device = device

    def push(self, data):
        self.buffer.append(data)

    def pull(self, n):
        minibatch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, ss_lst, o_lst = [], [], [], [], []

        for data in minibatch:
            s, a, r, ss, o = data
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            ss_lst.append(ss)
            o_lst.append(o)

        return s_lst, a_lst, r_lst, ss_lst, o_lst

    def size(self):
        return len(self.buffer)
