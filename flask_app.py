import os 
import numpy as np 
import cv2 
import flask 
import base64
from flask import Flask, request, jsonify
from file_similar import get_result, get_result_as_row
from utils.utils.utils import random_name, covert_img2base64, find_have_in_database, update_scores
from csv import writer
import pandas as pd

FILE_RESULT = 'utils/result/result.csv'

FILE_WEIGHT = 'utils/result/weight.csv'

def append_list_as_row(file_name, list_of_elem):
    
    with open(file_name, 'a') as write_obj:
        
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


app = Flask(__name__)

@app.route('/', methods = ['GET'])
def hello():

    return flask.render_template("home.html")
        
@app.route("/search", methods = ['POST'])
def search():
    if request.method == "POST":

        res = request.form['img64']
        # print(res)
        if res is None:
            return flask.jsonify({'error': 'no file'})
        
        imgString = res.split(',')[1]
        imgdata = base64.b64decode(imgString)
        os.remove("temp.jpg")
        with open('temp.jpg', 'wb') as f:
            f.write(imgdata)
        filename = find_have_in_database()
        
        print(filename)
        if filename is None:
            
            filename = 'upload/'+ random_name()
            with open(filename, 'wb') as f:
                f.write(imgdata)

            labels, list_img = get_result(filename)

            list_img_save = ['0'] + [filename] + list_img
            
            list_weight = ['0'] + [(len(list_img) - i) for i in range(len(list_img))]

            list_img = [os.path.join('dataset', labels, file) for file in list_img]

            list_img_str = [covert_img2base64(i) for i in list_img]

            # save in database result
           
            append_list_as_row(FILE_RESULT, list_img_save)
            append_list_as_row(FILE_WEIGHT, list_weight)

            index = [i for i in range(18)]
            data = {}

            for i, img_str  in enumerate(list_img_str):

                data[str(i+1)] = img_str
                data['id_img' + str(i+1)] =  str(index[i])

            
            data['source'] = filename
            data['status'] = 'success'
            
            return jsonify(data)
        else:

            # load result before to return.
            list_img, index = get_result_as_row(filename)
            
            # print(list_img)
            list_img = [str(i) for i in list_img]
            
            folder = 'dataset/'+  list_img[0].split('_')[0] 

            list_img = [os.path.join(folder, i) for i in list_img if i != 'nan']

            list_img_str = [covert_img2base64(i) for i in list_img]

            data = {}

            for i, img_str  in enumerate(list_img_str):

                data[str(i+1)] = img_str
                data['id_img' + str(i+1)] =  str(index[i])
            
            data['source'] = 'upload/' + filename
            data['status'] = 'success'
            return jsonify(data)

@app.route("/update", methods = ['POST'])
def update():     
    if request.method == "POST":
        img_raw = request.form['img']
        id_update =  request.form['update']
        
        print(img_raw)
        print(id_update)
        update_scores(img_raw, id_update)

        data = {}

        data['status'] = 'success'
        return data

if __name__ == "__main__":

    app.run()
