"""
delays.py
Collection of delays for algorithmic reverbs
"""
from collections import deque
import numpy as np


class Delay:
    """Circular buffer delay"""
    def __init__(self, sample_rate, delay_ms, delay_samples=None):
        if delay_samples:
            self.delay_length = delay_samples
        else:
            self.delay_length = int(delay_ms * (sample_rate / 1000.0))
        self.delay_line = deque(np.zeros(self.delay_length), self.delay_length)
        self.output = 0.0

    def tick(self, sample):
        """Run delay for 1 sample"""
        self.output = self.delay_line[-1]
        self.delay_line.appendleft(sample)

    def clear(self):
        """Flush delay line, setting all values to 0.0"""
        for _index in range(self.delay_length + 1):
            self.delay_line.appendleft(0.0)


class CombFilter:
    """Comb filter"""
    def __init__(self, sample_rate, delay_ms, coeff):
        self.delay = Delay(sample_rate, delay_ms)
        self.coeff = coeff
        self.output = 0.0

    def tick(self, sample):
        """Run delay for 1 sample"""
        delay_input = sample + self.coeff * self.delay.output
        self.output = self.delay.output

        self.delay.tick(delay_input)

    def clear(self):
        """Flush delay line, setting all values to 0.0"""
        self.delay.clear()


class AllPassDelay(CombFilter):
    """All pass delay"""
    def tick(self, sample):
        """Run delay for 1 sample"""
        delay_input = sample + self.coeff * self.delay.output
        self.output = ((-1.0 * self.coeff * delay_input) +
                       ((1.0 - self.coeff**2) * self.delay.output))

        self.delay.tick(delay_input)
