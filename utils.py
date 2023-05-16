import torch
import utils
# display = utils.notebook_init()
import subprocess
import cv2
import pandas as pd
import numpy as np
import os

#input = "C:\Users\saksh\data298\deploy\static\epch.png"

def panels(image):
    # img = cv2.imread(image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(img, (640, 640))
    command = ['python', 'yolov5/detect.py', '--save-crop', '--save-txt', '--source', image, '--weights', 'best_yolov5X.pt']
    result = subprocess.run(command, stdout=subprocess.PIPE)

    # Storing the panels and sorting them in correct order
    panels = []
    for root,dir,files in os.walk(os.path.abspath("yolov5/runs/detect/exp/crops/Panels/")):
        for file in files:
            panels.append((os.path.join(root, file)))
    panels.sort()
#Speech
    # Characters
    # Saving the croped dialouge images with texts data using Yolov5
    for i in range(len(panels)):
        command = ['python', 'yolov5/detect.py', '--save-crop', '--save-txt', '--source',
                   panels[i], '--weights', 'best_speechballoons.pt']
        result = subprocess.run(command, stdout=subprocess.PIPE)

    # This function is for sorting the data
    def sorting_speech_balloons(df, panel_number):
        sx = df.values.tolist()
        sx.sort(key=lambda x: x[2])
        # print(sx)
        x = []
        y = []
        k = 0
        for i in range(len(sx)):
            if sx[i][2] < sx[k][2] + 0.1:
                x.append(sx[i])
                # print(x)
            else:
                x.sort(key=lambda x: x[1])
                for j in range(len(x)):
                    y.append(x[j])
                # print(y)
                x = []
                # print(x)
                x.append(sx[i])
                k = i
        x.sort(key=lambda x: x[1])
        print("last", x)
        for p in range(len(x)):
            y.append(x[p])
        dff = pd.DataFrame(y)
        dff.columns = ["Class", "X_center", "y_center", "height", "width"]
        dff.insert(0, 'Panel_no', panel_number)
        return dff

    # Creating a dataframe from a single page (with different panels)
    # Speech_balloon table with distance
    speech_baloons_paths_main = []
    temp = []
    panel_number = 0
    dataframe = pd.DataFrame(columns=['Panel_no', 'Class', 'X_center', 'y_center', 'height', 'width'])
    for i in range(2, len(panels) + 2):
        panel_number += 1
        for root, dir, files in os.walk(os.path.abspath("yolov5/runs/detect/exp{0}/labels/".format(i))):
            for file in files:
                temp.append(os.path.join(root, file))
                print('sddddddddddddd', temp)
                df = pd.read_csv(temp[-1], header=None, sep=" ")
                dff = sorting_speech_balloons(df, panel_number)
                # print(dff)
                dataframe = pd.concat([dataframe,dff], ignore_index=True)

    def detect_text(results):
        """Detects text in the file."""
        from google.cloud import vision
        from google.oauth2 import service_account
        import io
        credentials = service_account.Credentials.from_service_account_file('Vison_API_key.json')
        client = vision.ImageAnnotatorClient(credentials=credentials)

        with io.open(results, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations
        # print('Texts:')
        # print(texts)

        for text in texts:
            # print('\n"{}"'.format(text.description))
            x = text.description
            detected = x.replace('\n', ' ')
            detected = detected.replace('\n', ' ')
            return detected
        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))

    speech_baloons_paths = []
    temp = []
    for i in range(2, len(panels) + 2):
        for root, dir, files in os.walk(os.path.abspath("yolov5/runs/detect/exp{0}/crops/".format(i))):
            for file in files:
                temp.append(os.path.join(root, file))
                # print(temp)
        temp.sort()
        speech_baloons_paths.append(temp)
        temp = []

    print(speech_baloons_paths)

    actual_text = []
    for i in range(len(speech_baloons_paths)):
        for j in range(len(speech_baloons_paths[i])):
            detected_text = detect_text(speech_baloons_paths[i][j])
            actual_text.append(detected_text)

    print(actual_text)

    text_dataframe = pd.DataFrame(actual_text, columns=['Text'])
    print(text_dataframe)

    Text_dataframe = dataframe.join(text_dataframe['Text'])

    # Joining both speech location data and its text into a single DataFrame
    print(Text_dataframe)

    # Characters

    for i in range(len(panels)):
        command = ['python', 'yolov5/detect.py', '--save-crop', '--save-txt', '--source',
                   panels[i], '--weights', 'best_char.pt']
        result = subprocess.run(command, stdout=subprocess.PIPE)

    # Speech_balloon table with distance
    # speech_baloons_paths_main = []
    temp = []
    panel_number = 0
    dataframe = pd.DataFrame(columns=['Panel_no', 'Class', 'X_center', 'y_center', 'height', 'width'])
    for i in range(len(panels) + 2, len(panels) + 2 + len(panels)):
        panel_number += 1
        for root, dir, files in os.walk(os.path.abspath("yolov5/runs/detect/exp{0}/labels/".format(i))):
            for file in files:
                temp.append(os.path.join(root, file))
                print('sddddddddddddd', temp)
                df = pd.read_csv(temp[-1], header=None, sep=" ")
                dff = sorting_speech_balloons(df, panel_number)
                # print(dff)
                dataframe = pd.concat([dataframe,dff], ignore_index=True)
    # concat
    #
    # print(dataframe)

    Character_table = dataframe

    # print(Character_table)
    #
    # print(Text_dataframe)

    '''
    Join the tables

    Right JOin (where right table is Speech)
    sqrt((x2-x1)2,(y2-y1)2) as distance
    Add a new column named rank based on distance
    Add a new column once the final table is ready. the column is numbered from 0 to n just to know the order
    Filter Class Blast and Class Narration into seperate table order by Speech_class
    Rest into other table with order by Char_class.'''

    JOined_data = Character_table.merge(Text_dataframe, left_on='Panel_no', right_on='Panel_no', how='right')

    JOined_data = JOined_data.drop(['height_x', 'width_x', 'height_y', 'width_y'], axis=1)
    JOined_data['Distance'] = np.sqrt(((JOined_data['X_center_y'] - JOined_data['X_center_x']) ** 2) + (
                (JOined_data['y_center_y'] - JOined_data['y_center_x']) ** 2))
    JOined_data['rnk'] = JOined_data.groupby(['Panel_no', 'Text'])['Distance'].rank(method='min',
                                                                                    ascending=True).astype(float)
    Joined = JOined_data.query("rnk == 1 | rnk.isnull()", engine='python')
    # print(Joined)
    Joined = Joined.sort_values(by=['Panel_no', 'X_center_y'])


    Joined = Joined.fillna(value={'Class_x': 200})
    return (Joined, panels)