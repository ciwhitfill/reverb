import pytest
import numpy as np
from scipy import signal, fft
from ..lib.delays import RingBuffer, Delay


def fill_noise(ring_buffer: RingBuffer):
    """Fill buffer with noise"""
    noise = np.random.rand(ring_buffer.length)
    for sample in noise:
        ring_buffer.push(sample)
    return noise


class TestRingBuffer:
    """Tests for RingBuffer class"""

    buffer_sizes = [1, 3, 1024, 48000, 480000]

    @pytest.mark.parametrize("size", buffer_sizes)
    def test_ring_buffer_clear(self, size):
        """Test clear"""
        ring_buffer = RingBuffer(size)
        ring_buffer.clear()
        _ = fill_noise(ring_buffer)
        ring_buffer.clear()
        for index in range(size):
            assert ring_buffer[index] == 0.0

    @pytest.mark.parametrize("size", buffer_sizes)
    def test_ring_buffer_access(self, size):
        """Test __getitem__"""
        ring_buffer = RingBuffer(size)
        ring_buffer.clear()
        input_ = fill_noise(ring_buffer)
        for index in range(size):
            assert ring_buffer[index] == input_[(-1 * index) - 1]
        for neg_index in range(-1, (-1 * size) - 1, -1):
            assert input_[neg_index] == ring_buffer[neg_index]

    @pytest.mark.parametrize("size", buffer_sizes)
    def test_ring_buffer_overwrite(self, size):
        """Fill buffer twice and test __getitem__"""
        ring_buffer = RingBuffer(size)
        ring_buffer.clear()
        _ = fill_noise(ring_buffer)
        input_ = fill_noise(ring_buffer)
        for index in range(size):
            assert ring_buffer[index] == input_[(-1 * index) - 1]
        for neg_index in range(-1, (-1 * size) - 1, -1):
            assert input_[neg_index] == ring_buffer[neg_index]


class TestDelay:
    """Test """

    delays = [
        (1, 1000, 'none', 0),
        (1, 1000, 'linear', 0),
        (1, 1000, 'hermite', 0),
        (24, 48000, 'none', 0)
    ]

    @pytest.mark.parametrize("time, sample_rate, interpolation, tolerance",
                             delays)
    def test_delay_length(self, time, sample_rate, interpolation, tolerance):
        """Test delay lengths"""
        sample_period: np.double = 1.0 / sample_rate
        delay = Delay(sample_rate, time * 2, [time], interpolation)
        samples = fft.next_fast_len(int(time * sample_rate))
        x = np.linspace(0.0, time, samples, endpoint=False)
        impulse = signal.unit_impulse(x.size)

        impulse_response = np.zeros(impulse.size)
        for index, sample in enumerate(impulse):
            delay.tick(sample)
            impulse_response[index] = delay.output

        samples_delayed = np.nonzero(impulse_response)[0]
        actual_delay_ms = (samples_delayed * sample_period) * 1000.0

        assert np.abs(actual_delay_ms - time) <= (tolerance * sample_period)
