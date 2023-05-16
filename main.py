import pickle
import re
from datetime import datetime
import os
from flask import Flask, render_template, request, session
import zipfile
import os
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
#lib for detection



from PIL import Image

import utils
from utils import panels
from utils2 import speech
from utils3 import result

app = Flask(__name__)

UPLOAD_FOLDER = "C:/Users/yalla/PycharmProjects/Comic_final1/static/"
UPLOAD_FOLDER1 = "C:/Users/yalla/PycharmProjects/Comic_final1/static/final/"
app.secret_key = 'This is your secret key to utilize session in Flask'


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    rem_image = []
    file_number = 1
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            print("This is",image_file.filename)
            if os.path.splitext(image_file.filename)[-1] == '.zip':
                image_location = os.path.join(UPLOAD_FOLDER1, image_file.filename)
                print(image_location)
                image_file.save(image_location)
                zip_file = zipfile.ZipFile(image_location)
                # destination_dir = UPLOAD_FOLDER
                zip_file.extractall(UPLOAD_FOLDER1)

                file_names = zip_file.namelist()
                print("The file names i guess", file_names)
                zip_file.close()
                print("File names", file_names)
                # for i in file_names:
                #     image_location = os.path.join(UPLOAD_FOLDER1, i)
                #     image_file.save(image_location)
                #     rem_image.append(image_location)
                #     print('rem_image is', rem_image)

            elif os.path.splitext(image_file.filename)[-1] == '.jpg' or os.path.splitext(image_file.filename)[-1] == '.png' or os.path.splitext(image_file.filename)[-1] == '.jpeg':
                image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
                image_file.save(image_location)
            # pred = panels(image_location)
            #print("The image location", image_location)
            if os.path.splitext(image_file.filename)[-1] == '.zip':
                extarcted_files = os.listdir(UPLOAD_FOLDER1)
                print(extarcted_files)
                for i in extarcted_files:
                    print("fire",i)
                    # print("FIre",file_names[i])
                    if i[-1] == 'p' or i[-1] == '4':
                        continue
                    else:
                        pred = panels(UPLOAD_FOLDER1 + i)
                        file_number += 1
                        session['pred_val'] = speech(pred,file_number)

                    # if i[-1] != 'p' or i[-1] != '4':
                    #     pred = panels(UPLOAD_FOLDER1+i)
                    # # pred = panels("static/"+file_names[i])  # give file path
                    #     session['pred_val'] = speech(pred)
                    # else:
                    #     continue
            else:
                pred = panels(image_location)  # give file path
                session['pred_val'] = speech(pred, file_number)
            result()
            # img_file_path = session.get('pred', None)
            return render_template('view.html')
            # return render_template("index.html", Prediction = pred)

            # #pred = predict_func(image_location, MODEL) #image path and model input
            # return render_template("index.html", Prediction = pred)
    return render_template("index.html", Prediction="result")


@app.route('/show_image')
def show_image():
    # img_file_path = session.get('pred_val', None)
    # Display image in Flask application web page
    # return render_template('display.html', user_image=img_file_path)
    return render_template('display.html')


if __name__ == "__main__":
    app.run(port=5000, debug=True)