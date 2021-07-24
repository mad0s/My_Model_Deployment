from logging import debug
from flask import Flask, render_template
from werkzeug.utils import html
from flask import Flask,render_template,request
import win32api
import tensorflow as tf
import numpy as np
import json
import os
import math
import librosa
from json import JSONEncoder


predictnew=""
Data="C:/Users/HTC/Desktop/FYP WEB/templates/input/input.wav"
JSON_PATH = "C:/Users/HTC/Desktop/FYP WEB/newfile.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
model=tf.keras.models.load_model('CNNmodel.h5')
app = Flask(__name__)
@app.route('/',methods=["GET","POST"])



def home():
    if request.method == "POST":
        print(request.form)
        filepath=request.files['file']
        filepath.save('C:/Users/HTC/Desktop/FYP WEB/templates/input/input.wav')
        prediction=save_mfcc(Data, JSON_PATH, num_segments=10)
        return render_template('show.html',predictnew=prediction)
    return render_template('home.html')


def back():
    if request.method == "POST":
        return render_template('home.html')

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    signal, sample_rate = librosa.load(dataset_path, sr=SAMPLE_RATE)
    for d in range(num_segments):
        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment
        # extract mfcc
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        if len(mfcc) == num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
            #print("{}, segment:{}".format(file_path, d+1))
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    with open("newfile.json","r") as fp:
        predictionarray=json.load(fp)
        newarray = np.array(predictionarray["mfcc"])
        #print(newarray[5].shape)
        newarray1=newarray[5]
        #print(newarray1)
        #print(newarray1.shape)
        newarray2 = newarray1[..., np.newaxis]
        #print(newarray2.shape)
        newarray3=newarray2[np.newaxis,...]
        #print(newarray3.shape)
        predict23=model.predict(newarray3)
        predictnew = np.argmax(predict23, axis=1)
        
        if predictnew==0:
            return "Blues"
        if predictnew==1:
            return "Classical"
        if predictnew==2:
            return "Country"
        if predictnew==3:
            return "Disco"
        if predictnew==4:
            return "HipHop"
        if predictnew==5:
            return "Jazz"
        if predictnew==6:
            return "Metal"
        if predictnew==7:
            return "Pop"
        if predictnew==8:
            return "Reggae"
        if predictnew==9:
            return "Rock"
        return "Genre Not Found"
    

    


if __name__ == "__main__":
    app.run(debug=False)


