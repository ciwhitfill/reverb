#!python
#cython: language_level=3

"""
delays.py
Collection of delays for algorithmic reverbs
"""

from typing import Dict
import cython
import numpy as np


@cython.cclass
class RingBuffer:
    """Circular buffer"""
    buffer: np.double[:]
    buffer_view: np.double[:, ::1]
    length: np.uintc
    write_pointer: np.uintc
    read_pointer: np.uintc

    def __init__(self, length: np.uintc) -> None:
        self.buffer = np.zeros(length, dtype=np.double)
        self.buffer_view = self.buffer
        self.length = length
        self.write_pointer = 0
        self.read_pointer = 1

    @cython.ccall
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def push(self, input_: np.double):
        """Push sample to delay line"""
        self.buffer_view[self.write_pointer] = input_
        self.write_pointer = (self.write_pointer + 1) % self.length
        self.read_pointer = (self.write_pointer + 1) % self.length

    @cython.ccall
    def clear(self):
        """Flush delay line, setting all values to 0.0"""
        for _index in range(self.length):
            self.push(0.0)

    def __getitem__(self, index: int) -> np.double:
        if index < 0:
            return self.buffer_view[(self.write_pointer - index) % self.length]

        return self.buffer_view[(self.read_pointer + index) % self.length]


@cython.cclass
class Delay:
    """Circular buffer delay"""
    stereo: bool
    output: np.double
    delay_length: np.uintc
    delay_line: RingBuffer

    def __init__(self, sample_rate: np.uintc, delay_ms: np.float,
                 delay_samples: np.uintc = 0):
        self.stereo = False
        self.output = 0.0
        if delay_samples > 0:
            self.delay_length = delay_samples
        else:
            self.delay_length = int(delay_ms * (sample_rate / 1000.0))

        self.delay_line = RingBuffer(self.delay_length)
        self.clear()

    def __getattr__(self, output):
        return self.output

    @cython.ccall
    def tick(self, input_):
        """Run delay for 1 sample"""
        self.output = self.delay_line[-1]
        self.delay_line.push(input_)

    @cython.ccall
    def clear(self):
        """Flush delay line, setting all values to 0.0"""
        for _index in range(self.delay_length + 1):
            self.delay_line.push(0.0)


@cython.cclass
class CombFilter:
    """Comb filter"""
    delay: Delay
    coeff: np.uintc
    output: np.double

    def __init__(self, sample_rate: np.uintc, delay_ms: np.float,
                 coeff: np.uintc):
        self.delay = Delay(sample_rate, delay_ms)
        self.coeff = coeff
        self.output = 0.0

    @cython.ccall
    def tick(self, input_: np.double):
        """Run delay for 1 sample"""
        delay_input: np.double = input_ + self.coeff * self.delay.output
        self.output = self.delay.output
        self.delay.tick(delay_input)

    @cython.ccall
    def clear(self):
        """Flush delay line, setting all values to 0.0"""
        self.delay.clear()

    def __getattr__(self, output):
        return self.output


@cython.cclass
class AllPassDelay(CombFilter):
    """All pass delay"""
    delay: Delay
    coeff: np.uintc
    output: np.double

    @cython.ccall
    def tick(self, input_: np.double):
        """Run delay for 1 sample"""
        delay_input: np.double = input_ + self.coeff * self.delay.output
        self.output = ((-1.0 * self.coeff * delay_input) +
                       ((1.0 - self.coeff**2) * self.delay.output))

        self.delay.tick(delay_input)


@cython.cclass
class FeedbackLowPass():
    """For use in delay feedback paths"""
    delay: Delay
    damp: np.single
    output: np.double

    def __init__(self, sample_rate: np.uintc, dampening: np.single):
        self.delay = Delay(sample_rate, 0, delay_samples=1)
        self.damp = dampening
        self.output = 0.0

    @cython.ccall
    def tick(self, input_: np.double):
        """"Run filter one step"""
        self.output = ((self.damp * self.delay.output) +
                       ((1.0 - self.damp) * input_))
        self.delay.tick(self.output)

    @cython.ccall
    def clear(self):
        """Flush delay lines, setting all values to 0.0"""
        self.delay.clear()
        self.output = 0.0

    def __getattr__(self, output):
        return self.output


@cython.cclass
class LowpassFeedbackCombFilter():
    """Moorer comb filter with lowpass in feedback path"""
    delay: Delay
    lowpass: FeedbackLowPass
    fb_gain: np.single
    output: np.double

    def __init__(self, sample_rate: np.uintc, delay_ms: np.single,
                 coeffs: Dict):
        self.delay = Delay(sample_rate, delay_ms)
        self.lowpass = FeedbackLowPass(sample_rate, coeffs["dampening"])
        self.fb_gain = coeffs["feedback_gain"]
        self.output = 0.0

    @cython.ccall
    def tick(self, input_: np.double):
        """Run delay for 1 sample"""
        self.output = self.delay.output
        delay_input: np.double = input_ + self.fb_gain * self.lowpass.output
        self.lowpass.tick(self.output)
        self.delay.tick(delay_input)

    @cython.ccall
    def clear(self):
        """Flush delay lines, setting all values to 0.0"""
        self.delay.clear()
        self.lowpass.clear()
        self.output = 0.0

    def __getattr__(self, output):
        return self.output


@cython.cclass
class LowpassFeedbackAllPass(LowpassFeedbackCombFilter):
    """All pass delay with low pass in feedback path"""
    delay: Delay
    lowpass: FeedbackLowPass
    fb_gain: np.single
    output: np.double

    @cython.ccall
    def tick(self, input_: np.double):
        """Run delay for 1 sample"""
        delay_input: np.double = input_ + self.fb_gain * self.lowpass.output
        self.output = ((-1.0 * self.fb_gain * delay_input) +
                       (1.0 - self.fb_gain**2) * self.delay.output)

        self.lowpass.tick(self.delay.output * self.fb_gain)
        self.delay.tick(delay_input)

    def __getattr__(self, output) -> np.double:
        return self.output


@cython.cclass
class MultiTapDelay():
    """Multi-tap delay"""
    stereo: bool
    output: np.double
    delay_taps: np.uintc[:]
    delay_line: RingBuffer
    num_delay_taps: np.uintc
    delay_length: np.uintc

    def __init__(self, sample_rate: np.uintc, delays_ms: np.ndarray):
        self.stereo = False
        self.num_delay_taps = len(delays_ms)
        self.delay_taps = (delays_ms * (float(sample_rate) /
                                        1000.0)).astype(int)
        self.delay_length = max(self.delay_taps)
        self.delay_line = RingBuffer(self.delay_length)
        self.num_delay_taps = len(self.delay_taps)
        self.output = 0.0
        self.clear()

    @cython.ccall
    def tick(self, input_: np.double):
        """Run delay for 1 sample"""
        self.output = 0.0
        mult: np.double = 1 / float(self.num_delay_taps)
        for delay in range(self.num_delay_taps):
            self.output += self.delay_line[self.delay_taps[delay]] * mult

        self.delay_line.push(input_)

    def __getattr__(self, output):
        return self.output

    @cython.ccall
    def clear(self):
        """Flush delay line, setting all values to 0.0"""
        for _index in range(self.delay_length + 1):
            self.delay_line.push(0.0)
