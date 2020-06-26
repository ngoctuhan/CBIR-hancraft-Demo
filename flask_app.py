import os 
import numpy as np 
import cv2 
import flask 
import base64
from flask import Flask, request, jsonify
from file_similar import get_result
from utils.utils.utils import random_name, covert_img2base64
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
        filename = os.path.join( 'upload', random_name() )
        with open(filename, 'wb') as f:
            f.write(imgdata)
        
        labels, list_img = get_result(filename)

        list_img = [os.path.join('dataset', labels, file) for file in list_img]

        list_img_str = [covert_img2base64(i) for i in list_img]

        data = {}

        for i, img_str  in enumerate(list_img_str):

            data[str(i+1)] = img_str
        
        data['status'] = 'success'
        return jsonify(data)

if __name__ == "__main__":

    app.run()
