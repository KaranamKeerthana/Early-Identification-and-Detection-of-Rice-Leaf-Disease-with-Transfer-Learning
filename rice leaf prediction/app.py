# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:34:20 2020

@author: Krish Naik
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='leaf.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="leaf_blight"
    elif preds==1:
        preds="Brown spot"
    elif preds==2:
        preds="Healthy"
    elif preds==3:
        preds="Leaf blast"
           
    return preds


 
@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/abstract')
def abstract():
	return render_template('abstract.html')

@app.route('/analysis')
def analysis():
	return render_template('analysis.html')

@app.route('/leaf_blight')
def leaf_blight():
	return render_template('leaf_blight.html')


@app.route('/leaf_blightt')
def leaf_blightt():
	return render_template('leaf_blightt.html')

@app.route('/leaf_blast')
def leaf_blast():
	return render_template('leaf_blast.html')

@app.route('/leaf_blastt')
def leaf_blastt():
	return render_template('leaf_blastt.html')

@app.route('/brown_spot')
def brown_spot():
	return render_template('brown_spot.html')

@app.route('/brown_spott')
def brown_spott():
	return render_template('brown_spott.html')

@app.route('/healthy')
def healthy():
	return render_template('healthy.html')

@app.route('/healthyy')
def healthyy():
	return render_template('healthyy.html')

@app.route('/leaf_blight_tel')
def leaf_blight_tel():
	return render_template('leaf_blight_tel.html')

@app.route('/leaf_blast_tel')
def leaf_blast_tel():
	return render_template('leaf_blast_tel.html')

@app.route('/brown_spot_tel')
def brown_spot_tel():
	return render_template('brown_spot_tel.html')

@app.route('/healthy_tel')
def healthy_tel():
	return render_template('healthy_tel.html')

@app.route('/future')
def future():
	return render_template('future.html')    

@app.route('/login')
def login():
	return render_template('login.html')

@app.route('/chart')
def chart():
	return render_template('chart.html')


@app.route('/first', methods=['GET'])
def first():
    # Main page
    return render_template('first.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds)
        if(preds=="Brown spot"):
            return render_template('brown_spott.html')
        elif(preds=="leaf_blight"):
            return render_template('leaf_blightt.html')
        elif(preds=="Leaf blast"):
            return render_template('leaf_blastt.html')
        elif(preds=="Healthy"):
            return render_template('healthyy.html')

        
    return "Diseases Leaf , Yet to be classified ."



if __name__ == '__main__':
    app.run(port=5001,debug=True)
