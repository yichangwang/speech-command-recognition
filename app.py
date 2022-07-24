import model.model as model
from flask import Flask, render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


app = Flask(__name__, template_folder='Template')
Bootstrap(app)

if not os.path.isdir("static"):
    os.mkdir("static")


def inference(wavfile_path):
    pre_proc = model.PreProcessing(wavfile_path)
    x_mfcc = pre_proc.get_mfcc_from_signal()
    x_mfcc = x_mfcc[None, :, :, None]
    net = model.Model(x_mfcc)
    return net.inference()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            wav_path = os.path.join('static', uploaded_file.filename)
            uploaded_file.save(wav_path)
            class_name = inference(wav_path)
            result = {
                'class_name': class_name,
                'wav_path': wav_path,
            }
            return render_template('result.html', result=result)
    return render_template('index.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            wav_path = os.path.join('static', uploaded_file.filename)
            uploaded_file.save(wav_path)
            class_name = inference(wav_path)
            # print(class_name)
            return class_name


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
