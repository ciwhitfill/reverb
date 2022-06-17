"""
reverbs.py
Algorithmic reverb implementations
"""


class SchroederReverb:
    """
    Schroeder reverbator, as described in
    'Natural Sounding Artificial Reverberation', Fig. 5
    """

    def __init__(self, coeffs, delay_length, allpass_matrix, sample_rate):
        self.size = len(allpass_matrix)
        self.gain_1 = coeffs["gain_1"]
        self.gain_2 = coeffs["gain_2"]
        self.output = 0.0

        self.allpass_reverb = [AllPassDelay(sample_rate,
                               allpass_matrix[allpass][0],
                               allpass_matrix[allpass][1])
                               for allpass in range(self.size)]

        self.delay = Delay(sample_rate, delay_length)

    def tick(self, input_):
        """Run reverb for 1 sample"""
        self.delay.tick(input_ + self.gain_2 * self.allpass_reverb[-1].output)
        for index, allpass in enumerate(self.allpass_reverb):
            if index == 0:
                allpass.tick(self.delay.output)
            else:
                allpass.tick(self.allpass_reverb[index - 1].output)

        self.output = ((self.allpass_reverb[-1].output *
                       (1.0 - self.gain_1**2)) +
                       (input_ * self.gain_1 * -1.0))

    def clear(self):
        """Flush delay lines, setting all values to 0.0"""
        self.delay.clear()
        for allpass in self.allpass_reverb:
            allpass.clear()
