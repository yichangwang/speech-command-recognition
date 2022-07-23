# speech-command-recognition

## Introduction

The object of this project is to recognize input voice commands.

The TensorFlow model is trained on [Google voice commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands).

The training is done on Google colab, and the training notebook file is included in the model folder.
## Preparation
```
git clone git@github.com:yichangwang/speech-command-recognition.git
cd speech-command-recognition
unzip test_data.zip
pip install -r requirements.txt
```

## Run the model
After install the requirements, you can host the flask app through
`python app.py`, and unzip `test_data.zip`, which includes the data for testing. 
There are two ways to use the model for prediction.
1. Through browser:
   open the URL http://localhost:5000 and upload a wav file from `test_data` folder.
2. Send POST requests to Flask API using Python requests module: `python send_request.py`, it will randomly choose 3 examples from each test class for inference.
