import numpy as np
import random
from collections import namedtuple
from collections import deque
from abc import ABC, abstractmethod

Experience = namedtuple("Experience", ('state', 'action', 'reward', 'next_state', 'done'))


class Buffer(ABC):

    def __init__(self, buffer_size, batch_size):
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._buffer = []

    # getter and setter
    @property
    def buffer_size(self):
        return self._buffer_size

    @property
    def batch_size(self):
        return self._batch_size

    def __len__(self):
        return len(self._buffer)

    @abstractmethod
    def append(self, experience):
        pass

    @abstractmethod
    def sample(self):
        pass


class ExperienceReplayBuffer(Buffer):

    def __init__(self, buffer_size, batch_size):
        super(ExperienceReplayBuffer, self).__init__(buffer_size, batch_size)
        self._buffer = deque(maxlen=self._buffer_size)

    def append(self, experience):
        # if len(_buffer) = buffer_size, then .append kicks of first object in deque
        # Note from AC: clever use of deques, Khalil!
        self._buffer.append(experience)

    def sample(self):
        if len(self._buffer) > self._batch_size:
            buffer_sample = random.sample(self._buffer, self._batch_size)

        else:
            buffer_sample = random.sample(self._buffer, len(self._buffer))
        return buffer_sample