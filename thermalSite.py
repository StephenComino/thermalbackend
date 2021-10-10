from flask import Flask
from flask import g
from flask import Response
from flask import request
from flask import jsonify
from flask import render_template
from thermalCam import ThermalCamera
from depthcam import DepthCamera
from logData import logData
from gevent import monkey
import numpy as np
from numpyencoder import NumpyEncoder
import cv2
import json
import threading
import time
import os
import _pickle as pickle
import schedule
import os.path

DRAW_CONSTANT = 10
monkey.patch_all()
app = Flask(__name__)

outputFrame = None
lock = threading.Lock()
cam = ThermalCamera()
depth = DepthCamera()
schedule.every(1).minutes.do(depth.writeDepthDataToFile)


@app.route("/")
def index():
  agent = request.headers.get('User-Agent')
  with open('/home/pi/Desktop/flaskserver/debug/website_debug.txt', 'w+') as f:
      f.write('Connected from ' + agent + '\n')
      f.write('Loading History Data....\n')
      data = []
      if os.path.exists('/home/pi/Desktop/flaskserver/debug/requestData.pkl'):
          with open('/home/pi/Desktop/flaskserver/debug/requestData.pkl', 'rb') as fa:
              while True:
                  try:
                      data.append(pickle.load(fa))
                  except EOFError:
                      break
                  #data.append(pickle.load(fa))
      # Load history of data gathered
              f.write(str(data))
      else:
          data.append("hi")
      # Display History from File)
      # Let user delete this history from the website.
  return render_template(
      'home.html',
      datas=data)

@app.route("/video_stream", methods=['GET'])
def video_stream():
    type_stream = request.args.getlist('type', type=int)
    if type_stream[0] == 1:
        return Response(depth.getCameraView(),
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
    # Get the pixels drawn
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
    for x in range(160):
        for y in range(120):
            result = cv2.pointPolygonTest(contours, (idx_x,idx_y), False)
            if result == 1 or result == 0:
                points.append((x, y))
    
    # Pass points and get the minum and max value
    minimum, maximum = g.d_cam.getDepthData(points)
    
    return jsonify(isError= False,
                message= "Success",
                statusCode= 200,
                data= [minimum]), 200
    
    ## Need to resize and at the moment these are wrong
    #min_x_pixel = [(min_x // 2.34375) for min_x in min_x_pixel]
    #min_y_pixel = [(min_y // 2.5) for min_y in min_y_pixel]
# Test if this function highlights the correct area on the screen
# Draw the path on the 160x120 image

@app.route("/tempAt", methods = ['GET'])
def get_temp():
    min_x_pixel = request.args.getlist('x', type=float)
    min_y_pixel = request.args.getlist('y', type=float)
    
    log_data = logData()

    d_pixel_x = min_x_pixel.copy()
    d_pixel_y = min_y_pixel.copy()
    
    log_data.x_cords = d_pixel_x
    log_data.y_cords = d_pixel_y
    # Resize the image from the screen
    # 160 x 120 --> 375 x 299
    min_x_pixel = [(min_x // 2.34375) for min_x in min_x_pixel]
    min_y_pixel = [(min_y // 2.5) for min_y in min_y_pixel]
    # 160 x 120 --> 640 x 480
    min_x_depth_pixel = [int(min_x * 1.71) for min_x in d_pixel_x]
    min_y_depth_pixel = [int(min_y * 1.61) for min_y in d_pixel_y]
    new_contour_array = []
    for i,d in enumerate(min_x_pixel):
        arr = []
        arr.append(min_x_pixel[i])
        arr.append(min_y_pixel[i])
        new_contour_array.append(arr) 
    original_length = len(min_x_pixel)
    contours = np.array(new_contour_array, dtype=np.int32)
    
    # depth Contours
    new_contour_array_depth = []
    for i,d in enumerate(min_x_depth_pixel):
        arr = []
        arr.append(min_x_depth_pixel[i])
        arr.append(min_y_depth_pixel[i])
        new_contour_array_depth.append(arr) 
    original_length_depth = len(d_pixel_x)
    contours_depth = np.array(new_contour_array_depth, dtype=np.int32)
    #min_x_pixel.append(min_x_pixel[0])
    #min_y_pixel.append(min_y_pixel[0])
    data = cam.thermalData()
    
    # Test Here
    #cv2.drawContours(data,[contours],0,(255,255,255),2)
    #### ENDIND ################    
    #path = '/home/pi/Desktop/test.jpg'
    mask = np.zeros(data.shape, np.int8)
    cv2.drawContours(mask, [contours], -1, 255, -1)
    # mean = float(cv2.mean(data, mask=mask)[0] * 0.01 - 273.15)
    # Get the indices of mask where value == 255, which may be later used to slice the array.
    minimum = 100000
    maximum = 0
    mean = 0
    count = 1
    total = 0
    
    #mean = np.mean(data)
    #standard_deviation = np.std(data)
    #distance_from_mean = abs(data - mean)
    #max_deviations = 2
    #not_outlier = distance_from_mean < max_deviations * standard_deviation
    #no_outliers = data[not_outlier]
    
    distanceOfPoint = []
    for idx_x in range(640):
        for idx_y in range(480):
            result = cv2.pointPolygonTest(contours_depth, (idx_x,idx_y), False)
            if result == 1 or result == 0:
                distanceOfPoint.append((idx_x, idx_y))
    
    # Get distance of points
    
#         g.d_cam = DepthCamera()
    distance_list = depth.getDepthData(distanceOfPoint)
    
    max_distance = 0
    min_distance = 0
    items_points = []
    #distanceOfPoint = []
    for idx_x, x in enumerate(data):
        for idx_y, y in enumerate(data[idx_x]):
            result = cv2.pointPolygonTest(contours, (idx_x,idx_y), False)
            if result == 1 or result == 0:
                cv2.circle(mask, (idx_x, idx_y), 1, (255,0,0), 1)                
                #if distance[idx_x, idx_y] > max_distance:
                #max_distance = distance[idx_x, idx_y]
                if data[idx_x, idx_y][0] > maximum:
                    temp = data[idx_x, idx_y][0] * 0.01 - 273.15
                    if data[idx_x, idx_y][0] and temp < 100:
                        maximum = data[idx_x, idx_y][0]
                if data[idx_x, idx_y][0] < minimum:
                    #if data[idx_x, idx_y][0] > 0:
                    temp = data[idx_x, idx_y][0] * 0.01 - 273.15 
                    if data[idx_x, idx_y][0] and temp >= -10:
                        minimum = data[idx_x, idx_y][0]
                
                temp = data[idx_x, idx_y][0] * 0.01 - 273.15
                if temp >= -10 and temp < 100:
                    total += temp
                    count += 1
                else:
                    continue
    
    mean = (total / count)
    
    #if minimum in no_outliers:
    minimum = minimum * 0.01 - 273.15
    #else:
        #minimum = 0
    #if maximum in no_outliers:
    maximum = maximum * 0.01 - 273.15
    #else:
        #maximum = 0;
    max_distance = max(distance_list)
    min_distance = min(distance_list)
    
    #with open('/home/pi/Desktop/file.txt', 'wb+') as file:
    #    file.write(pickle.dumps(distance))
    #cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX) # extend contrast
    #np.right_shift(data, 8, data) # fit data into 8 bits
   
    if minimum < 0:
        minimum = 0
    if maximum < 0:
        maximum = 0
    if mean < 0:
        mean = 0
    #final_temp = [minimum_temp, maximum_temp, avg_temp]
    final_temp = [minimum, maximum, mean, float(max_distance)]
    logData.max_temp = maximum
    logData.min_temp = minimum
    logData.avg_temp = mean
    logData.distance = max_distance
    save_data = {'max_temp': maximum, 'min_temp': minimum, 'avg_temp': mean, 'distance': max_distance }
    with open('/home/pi/Desktop/flaskserver/debug/requestData.pkl', 'ab+') as f:
        pickle.dump(save_data, f)
    #json_dump = json.dumps(data, cls=NumpyEncoder)
    
    return jsonify(isError= False,
                message= "Success",
                statusCode= 200,
                data= final_temp), 200
    
if __name__ == "__main__":
  #app.config['d_cam'] = DepthCamera()
    depth.writeDepthDataToFile()
    app.run(host='0.0.0.0',debug=False)
#     g.d_cam = DepthCamera()
