from flask import Flask
from flask import Response
from flask import request
from flask import jsonify
from test import ThermalCamera
from depthcam import getDepthData, getCameraView
from gevent import monkey
import numpy as np
from numpyencoder import NumpyEncoder
import cv2
import json
import threading

DRAW_CONSTANT = 10
monkey.patch_all()
app = Flask(__name__)

outputFrame = None
lock = threading.Lock()
cam = ThermalCamera()

@app.route("/")
def index():
  return "<html><body><h1>I Love my Darling Stephanie</h1></body></html>"

@app.route("/video_stream", methods=['GET'])
def video_stream():
    type_stream = request.args.getlist('type', type=int)
    if type_stream[0] == 1:
        return Response(getCameraView(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")
    elif type_stream[0] == 0:
        return Response(cam.cameraCapture(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/reset", methods = ['GET'])
def get_reset():
    cam.cameraReset()
    return jsonify(isError= False,
            message= "Success",
            statusCode= 200,
            data= "Ok"), 200

@app.route("/distanceAt", methods=['GET'])
def get_distance():
    min_x_pixel = request.args.getlist('x', type=float)
    min_y_pixel = request.args.getlist('y', type=float)
    
    new_contour_array = []
    for i,d in enumerate(min_x_pixel):
        arr = []
        arr.append(min_x_pixel[i])
        arr.append(min_y_pixel[i])
        new_contour_array.append(arr) 
    original_length = len(min_x_pixel)
    contours = np.array(new_contour_array, dtype=np.int32)
    
    mask = np.zeros(data.shape, np.uint16)
    cv2.drawContours(mask, [contours], -1, 255, -1)
    points = []
    for x in range(640):
        for y in range(480):
            result = cv2.pointPolygonTest(contours, (idx_x,idx_y), False)
            if result == 1 or result == 0:
                points.append((x, y))
                
    data = getDepthData(points)
    
    return jsonify(isError= False,
                message= "Success",
                statusCode= 200,
                data= [data]), 200
    
    ## Need to resize and at the moment these are wrong
    #min_x_pixel = [(min_x // 2.34375) for min_x in min_x_pixel]
    #min_y_pixel = [(min_y // 2.5) for min_y in min_y_pixel]
# Test if this function highlights the correct area on the screen
# Draw the path on the 160x120 image

@app.route("/tempAt", methods = ['GET'])
def get_temp():
    min_x_pixel = request.args.getlist('x', type=float)
    min_y_pixel = request.args.getlist('y', type=float)
    # Resize the image from the screen
    # 160 x 120 --> 375 x 299
    min_x_pixel = [(min_x // 2.34375) for min_x in min_x_pixel]
    min_y_pixel = [(min_y // 2.5) for min_y in min_y_pixel]
    
    new_contour_array = []
    for i,d in enumerate(min_x_pixel):
        arr = []
        arr.append(min_x_pixel[i])
        arr.append(min_y_pixel[i])
        new_contour_array.append(arr) 
    original_length = len(min_x_pixel)
    contours = np.array(new_contour_array, dtype=np.int32)
    #min_x_pixel.append(min_x_pixel[0])
    #min_y_pixel.append(min_y_pixel[0])
    data = cam.thermalData()
    # Test Here
    #cv2.drawContours(data,[contours],0,(255,255,255),2)
    #### ENDIND ################    
    #path = '/home/pi/Desktop/test.jpg'
    mask = np.zeros(data.shape, np.uint16)
    cv2.drawContours(mask, [contours], -1, 255, -1)
    # mean = float(cv2.mean(data, mask=mask)[0] * 0.01 - 273.15)
    # Get the indices of mask where value == 255, which may be later used to slice the array.
    minimum = 100000
    maximum = 0
    mean = 0
    count = 0
    total = 0
    
    mean = np.mean(data)
    standard_deviation = np.std(data)
    distance_from_mean = abs(data - mean)
    max_deviations = 2
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    no_outliers = data[not_outlier]
    
    for idx_x, x in enumerate(data):
        for idx_y, y in enumerate(data[idx_x]):
            result = cv2.pointPolygonTest(contours, (idx_x,idx_y), False)
            if result == 1 or result == 0:
                if data[idx_x, idx_y][0] > maximum:
                    if data[idx_x, idx_y][0] in no_outliers:
                        maximum = data[idx_x, idx_y][0]
                if data[idx_x, idx_y][0] < minimum:
                    #if data[idx_x, idx_y][0] > 0:
                    temp = data[idx_x, idx_y][0] * 0.01 - 273.15 
                    if data[idx_x, idx_y][0] in no_outliers and temp >= -10:
                        minimum = data[idx_x, idx_y][0]
                count += 1
                temp = data[idx_x, idx_y][0] * 0.01 - 273.15
                if temp >= -10:
                    total +=  data[idx_x, idx_y][0]
                else:
                    continue
    
    mean = (total / count) * 0.01 - 273.15
    
    #if minimum in no_outliers:
    minimum = minimum * 0.01 - 273.15
    #else:
        #minimum = 0
    #if maximum in no_outliers:
    maximum = maximum * 0.01 - 273.15
    #else:
        #maximum = 0;
    #path = '/home/pi/Desktop/test.jpg'
    #cv2.imwrite(path, mask)
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX) # extend contrast
    np.right_shift(data, 8, data) # fit data into 8 bits
   
    
    #final_temp = [minimum_temp, maximum_temp, avg_temp]
    final_temp = [minimum, maximum, mean]
    #json_dump = json.dumps(data, cls=NumpyEncoder)
    
    return jsonify(isError= False,
                message= "Success",
                statusCode= 200,
                data= final_temp), 200
    
if __name__ == "__main__":
  app.run(host='0.0.0.0',debug=False)
