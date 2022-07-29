import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import numpy as np
from tensorflow import keras
from utils.pre_processing import PreProcessing

# label encoding
with open('model/encoder_mapping.pkl', 'rb') as f:
    encoder_mapping = pickle.load(f)


class Model():
    def __init__(self, x_mfcc):
        self.x_mfcc = x_mfcc

    def inference(self):
        net = keras.models.load_model("model/saved_model.h5")
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
