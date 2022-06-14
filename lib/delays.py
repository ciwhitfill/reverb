# distutils: language = c++

import numpy as np
import cython


@cython.cclass
class RingBuffer:
    """Circular buffer"""
    buffer: np.double[:]
    buffer_view: np.double[:, ::1]
    length: np.uintc
    pointer: np.uintc

    def __cinit__(self, length: np.uintc) -> None:
        self.buffer = np.zeros(length, dtype=np.double)
        self.buffer_view = self.buffer
        self.length = length
        self.pointer = 0

    @cython.ccall
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def push(self, input_: np.double):
        """Push sample to delay line"""
        self.buffer_view[self.pointer] = input_
        self.pointer = (self.pointer + 1) % self.length

    @cython.ccall
    def clear(self):
        """Flush delay line, setting all values to 0.0"""
        for _index in range(self.length):
            self.push(0.0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __getitem__(self, index: np.uintc) -> np.double:
        if index < 0:
            return self.buffer_view[(self.pointer + index) % self.length]

        return self.buffer_view[(self.pointer - index - 1) % self.length]


@cython.cclass
class Delay:
    """Circular buffer delay"""
    interpolation: bool
    output: np.double
    delay_length: np.uintc
    delay_buffer: RingBuffer
    sample_rate: np.double
    delay_taps: np.double[:]

    def __cinit__(self, sample_rate: np.uintc, max_delay_ms: np.double, delay_taps, interpolation: str):
        self.interpolation = interpolation
        self.sample_rate = sample_rate
        self.output = 0.0
        self.delay_taps = delay_taps

        self.delay_length = int(max_delay_ms * (sample_rate / 1000.0) + 2)
        self.delay_buffer = RingBuffer(self.delay_length)
        self.clear()

    def __getattr__(self, output):
        return self.output

    @cython.ccall
    def tick(self, input_: np.double):
        """Run delay for 1 sample"""
        self.output = 0
        for tap in self.delay_taps:
            self.output += self.read(tap)
        self.delay_buffer.push(input_)

    @cython.ccall
    def read(self, delay_tap: np.double) -> cython.double:
        delay: np.double = delay_tap * (self.sample_rate / 1000.0)
        return self.interpolate(delay)

    @cython.ccall
    def interpolate(self, x_bar: np.double) -> cython.double:
        """Interpolate between two points in the delay line"""
        floor: np.uintc = int(np.floor(x_bar))
        remainder: np.uintc = x_bar - floor
        y1: np.double = self.delay_buffer[floor]
        y2: np.double = self.delay_buffer[floor + 1]

        if self.interpolation == 'linear':
            interpolate: np.double = (y2 - y1) * (remainder)
            return y1 + interpolate

        elif self.interpolation == 'hermite':
            y0: np.double = self.delay_buffer[floor - 1]
            y3: np.double = self.delay_buffer[floor + 2]
            slope0: np.double = (y2 - y0) * 0.5
            slope1: np.double = (y3 - y1) * 0.5
            v: np.double = y1 - y2
            w: np.double = slope0 + v
            a: np.double = w + v + slope1
            b_neg: np.double = w + a
            stage1: np.double = a * remainder - b_neg
            stage2: np.double = stage1 * remainder + slope0
            return stage2 * remainder + y1

        elif self.interpolation == 'none':
            nearest_neighbor: np.uintc = int(np.round(x_bar))
            return self.delay_buffer[nearest_neighbor]

    @cython.ccall
    def clear(self):
        """Flush delay line, setting all values to 0.0"""
        for _index in range(self.delay_length + 1):
            self.delay_buffer.push(0.0)


@cython.cclass
class CombFilter:
    """Comb filter"""
    delay: Delay
    coeff: np.uintc
    output: np.double

    def __cinit__(self, sample_rate: np.uintc, delay_ms: float, coeff: np.uintc):
        self.delay = Delay(sample_rate, delay_ms, [delay_ms], 'none')
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
class SchroederAllPass(CombFilter):
    """All pass delay"""
    delay: Delay
    coeff: np.uintc
    output: np.double

    @cython.ccall
    def tick(self, input_: np.double):
        """Run delay for 1 sample"""
        delay_input: np.double = input_ + self.coeff * self.delay.output
        self.output = (-1.0 * self.coeff * input_) + \
            ((1.0 - (self.coeff**2)) * self.delay.output)

        self.delay.tick(delay_input)
