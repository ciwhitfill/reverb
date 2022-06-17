"""
utils.py
Micellaneous helper functions
"""
from scipy.io import wavfile
from scipy import fft
import numpy as np

import IPython
from IPython import display


def process_file(input_path, output_path, processor):
    """"Process wav file with processor"""
    sample_rate, input_ = wavfile.read(input_path)
    scale_factor = 2**15

    output = np.zeros(fft.next_fast_len(input_.size))

    for index, sample in enumerate(input_):
        processor.tick(float(sample)/(scale_factor))
        output[index] = processor.output * scale_factor

    wavfile.write(output_path, sample_rate, output.astype(np.int16))
