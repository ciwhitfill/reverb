"""
utils.py
Micellaneous helper functions
"""
from scipy.io import wavfile
import numpy as np

import IPython
from IPython import display


def process_file(input_path, output_path, processor):
    """"Process wav file with processor"""
    sample_rate, input_ = wavfile.read(input_path)
    scale_factor = 2**15

    if processor.stereo:
        output = np.zeros((input_.size, 2))
    else:
        output = np.zeros(input_.size)

    for index, sample in enumerate(input_):
        processor.tick(float(sample)/(scale_factor))
        if processor.stereo:
            output[index][0] = processor.output[0] * scale_factor
            output[index][1] = processor.output[1] * scale_factor
        else:
            output[index] = processor.output * scale_factor

    wavfile.write(output_path, sample_rate, output.astype(np.int16))
