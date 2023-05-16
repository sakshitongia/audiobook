import pickle
import re
from datetime import datetime
import os
from flask import Flask, render_template, request, session

import os
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
#lib for detection



from PIL import Image

import utils 
from utils import predict


app = Flask(__name__)

UPLOAD_FOLDER = "C:/Users/saksh/data298/application/static"
# model = pickle.load(open('C:/Users/saksh/data298/deploy/models/model.pkl','rb'))

app.secret_key = 'This is your secret key to utilize session in Flask'


# def displayImage():
#             # Upload file flask
#         uploaded_img = request.files['uploaded-file']
#         # Extracting uploaded data file name
#         img_filename = secure_filename(uploaded_img.filename)
#         # Upload file to database (defined uploaded folder in static path)
#         uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
#         # Storing uploaded file path in flask session
#         session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
 


#     # Retrieving uploaded file path from session
#     img_file_path = session.get('uploaded_img_file_path', None)
#     # Display image in Flask application web page
#     return render_template('show_image.html', user_image = img_file_path)
 


@app.route("/", methods = ["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)

            #pred = panels(image_location)
            session['pred_val'] = predict(image_location) #give file path
            #img_file_path = session.get('pred', None)
            return render_template('view.html')
            #return render_template("index.html", Prediction = pred)


            # #pred = predict_func(image_location, MODEL) #image path and model input
            # return render_template("index.html", Prediction = pred)
    return render_template("index.html", Prediction = "result")

@app.route('/show_image')
def show_image():
    img_file_path = session.get('pred_val', None)
    # Display image in Flask application web page
    return render_template('display.html', user_image = img_file_path)
 
if __name__=='__main__':
    app.run(debug = True)



# Panels -> panel files
# panel files -> character prediction code -> coordinates 
# panel file -> speech bubbles prediction -> coordinates 
# connect character with speech bubble
# update dataframe 
# 




if __name__ == "__main__":
    app.run(port = 5000, debug = True)
    #app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))