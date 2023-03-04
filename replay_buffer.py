"""
ReplayBuffer class file.
"""

from collections import deque
import random


class ReplayBuffer():
    """
    Class of replay buffer.
    This replay buffer includes functions such as push and pull.
    Each data consists of state, action, reward, next_state, o.
    (if o is 0, then s' is terminal state.)
    """
    def __init__(self, device, max_size):
        self.buffer = deque(maxlen=max_size)
        self.device = device

    def push(self, data):
        """
        Push one set of data into buffer.

        :arg data:
            data should be a tuple of state, action, reward, next_state, o.
            The variable is 0 if the next_state is terminal state, or 1.
        """
        self.buffer.append(data)

    def pull(self, data_size):
        """
        Pull data of size data_size from buffer.

        :arg data_size:
            The size of data which will be pulled from buffer.

        :return:
            A tuple which consists of lists of state, action, reward,
            next_state, o.
        """
        minibatch = random.sample(self.buffer, data_size)
        s_lst, a_lst, r_lst, ss_lst, o_lst = [], [], [], [], []

        for data in minibatch:
            state, action, reward, next_state, o_value = data
            s_lst.append(state)
            a_lst.append(action)
            r_lst.append(reward)
            ss_lst.append(next_state)
            o_lst.append(o_value)

        return s_lst, a_lst, r_lst, ss_lst, o_lst

    def size(self):
        """
        Returns the size of buffer.
        """
        return len(self.buffer)
