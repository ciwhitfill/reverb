import pytest
import numpy as np
from ..lib.delays import RingBuffer


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
