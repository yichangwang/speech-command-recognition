import os
import random
import requests

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str,
                    default='127.0.0.1:5000',
                    help='ip:port')
parser.add_argument('--repeat', type=int, default=3)

args = parser.parse_args()

url = 'http://' + args.host + '/predict_api'

input_data_folder = 'test_data'
wav_list = os.listdir(input_data_folder)
test_folder = 'test_data'
y_folder_list = os.listdir(test_folder)

for y_true in y_folder_list:
    print(f"will test for: {y_true}")
    sample_folder = os.path.join(test_folder, y_true)
    test_sample = os.listdir(sample_folder)

    # randomly choose N (here = 3) from a random label
    file_names = random.sample(test_sample, args.repeat)
    for file_name in file_names:
        wavfile_path = os.path.join(sample_folder, file_name)

        with open(wavfile_path, 'rb') as f:
            my_file = {'file': f}
            values = {'upload_file': wavfile_path}
            r = requests.post(url, files=my_file, data=values)
        print(f"The prediction of {file_name} is {r.text}")
