# speech-command-recognition

## Introduction

The object of this project is to recognize input voice commands.

The TensorFlow model is trained on [Google voice commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands).

The training is done on Google colab, and the training notebook file is included in the model folder.

The model is encapsulated in docker container as a Flask web application.
Once deployed, we can run inference through web browser or python requests.
## Preparation
```
git clone git@github.com:yichangwang/speech-command-recognition.git
cd speech-command-recognition
unzip test_data.zip
pip install -r requirements.txt
```

## Run the model
After install the requirements, you can host the flask app through
`python flask_app.py`.
There are two ways to use the model for prediction.
1. Through browser:
   open the URL http://localhost:5000 and upload a wav file from `test_data` folder.
2. Send POST requests to Flask API using Python requests module: `python send_request.py` (note that you need to open another terminal window), it will randomly choose 3 examples from each test class for inference.

## Docker image
### Build by yourself
Please note that our Dockerfile uses tensorflow/tensorflow as base image,
there's no need to download some requirements.
You can remove `tensorflow`, `numpy`, and `scipy` from the file `requirements.txt` before run the commands below.
```
docker build -t speech-command .
docker run -p 5000:5000 speech-command
```
### Deploy
The build image is pushed to docker hub, to run the container, just run
```
docker pull yichangwang/speech-command
docker run yichangwang/speech-command
```