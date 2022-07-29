import random
from scipy.io import wavfile
import numpy as np
from sonopy import mfcc_spec


class PreProcessing():
    def __init__(self, input_data_path):
        self.input_data_path = input_data_path

    def normalize(self, audio_signal):
        return np.array(audio_signal) / max(np.absolute(audio_signal))

    def add_padding(self, audio_signal, desired_len):
        length_signal = len(audio_signal)
        if length_signal < desired_len:
            zeros = list(np.zeros(desired_len - length_signal))
            cut = random.randint(0, len(zeros))
            left = zeros[:cut]
            right = zeros[cut:]
            audio_signal = left + list(audio_signal) + right
        return audio_signal

    def get_signal(self, path):
        rate, sig = wavfile.read(filename=path)
        # sample rate == 16 000
        try:
            sig = sig[:, 0]
        except:
            pass
        if rate > 16000:
            sig = sig[::rate//16000+1]
        # normalization
        sig = self.normalize(sig)
        # standardization of sizes
        sig = self.add_padding(sig, 16000)

        return sig

    def get_mfcc_from_signal(self):
        signal = self.get_signal(self.input_data_path)
        # extract MFCCs features
        single_mfcc = mfcc_spec(signal, 16000, window_stride=(
            400, 160), fft_size=512, num_filt=20, num_coeffs=13).T
        # dropping first coefficient
        single_mfcc = single_mfcc[1:, :]  # keeping only 12 coefficients

        return single_mfcc
