from collections import deque
import random


class Replay_Buffer():
    """
    Class of replay buffer.
    This replay buffer has max size of 10000,
    and includes functions such as push and pull.
    Each data consists of s, a, r, s'.
    """
    def __init__(self, device):
        self.buffer = deque(maxlen=10000)
        self.device = device

    def push(self, data):
        self.buffer.append(data)

    def pull(self, n):
        minibatch = random.sample(self.buffer, n)
        s_lst, a_lst, v_lst, ss_lst = [], [], [], []

        for data in minibatch:
            s, a, v, ss = data
            s_lst.append(s)
            a_lst.append(a)
            v_lst.append(v)
            ss_lst.append(ss)

        return s_lst, a_lst, v_lst, ss_lst

    def size(self):
        return len(self.buffer)
