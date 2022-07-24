from tensorflow import keras
from sonopy import mfcc_spec
from scipy.io import wavfile
import numpy as np
import random
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# from sklearn import preprocessing

with open('model/encoder_mapping.pkl', 'rb') as f:
    encoder_mapping = pickle.load(f)


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


class Model():
    def __init__(self, x_mfcc):
        self.x_mfcc = x_mfcc

    def inference(self):
        net = keras.models.load_model("model/speech_recognition.h5")
        pred = net(self.x_mfcc)
        # pred = net.predict(self.x_mfcc)
        # this predict will trigger tf.function retracing.
        # https://stackoverflow.com/questions/66271988/warningtensorflow11-out-of-the-last-11-calls-to-triggered-tf-function-retracin
        pred = int(np.argmax(pred, axis=1))
        pred = encoder_mapping[pred]
        return pred


if __name__ == "__main__":
    import random
    test_folder = 'test_data'
    y_folder_list = os.listdir(test_folder)

    for y_true in y_folder_list:
        print(f"will test for: {y_true}")
        sample_folder = os.path.join(test_folder, y_true)
        test_sample = os.listdir(sample_folder)

        # randomly choose N (here = 3) from a random label
        file_names = random.sample(test_sample, 3)
        for file_name in file_names:
            wavfile_path = os.path.join(sample_folder, file_name)
            pre_proc = PreProcessing(wavfile_path)
            x_mfcc = pre_proc.get_mfcc_from_signal()
            x_mfcc = x_mfcc[None, :, :, None]
            model = Model(x_mfcc)
            print(f"{file_name} is classified as:", model.inference())
