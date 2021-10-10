## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#####################################################
## librealsense tutorial #1 - Accessing depth data ##
#####################################################

# First import the library
import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
import time
import schedule
from threading import Thread

class DepthCamera():
    def __init__(self):
        # Get the data of the camera
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        #self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
        
       
        # Enable the streams we are interested in
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        
        
        self.pipeline.start(self.config)
        self.frames = None
        self.Depth_Image = 0
        self.average = 0
        #schedule.every(1).minutes.do(self.writeDepthDataToFile)
        
        #s.enter(60, 1, writeDepthDataToFile, "")
    # Get Frame
    def getFrame(self):
        frames = self.pipeline.wait_for_frames()
        return frames
    # Get the RGB frame
    def getRGBFrame(self):
        frames = self.frames.get_color_frame()
        return frames
    
    # Get the Depth Frame
    def getDepthFrame(self, frames):
        depth_frame = frames.get_depth_frame()
        return depth_frame
    # Configure depth and color streaamems
    
    def getDepthData(self, image_points):

        depth_frame = np.load('/home/pi/Desktop/DepthData.npy')
        #if not depth_frame:
        #dim = (160, 120)
        distance_data = []
        for i, tu in enumerate(image_points):
            if tu[0] >= 480:
                distance_data.append(depth_frame[479][tu[1]])
            elif tu[1] >= 480:
                distance_data.append(depth_frame[tu[0]][tu[479]])
            elif tu[0] >= 480 and tu[1] >= 480:
                distance_data.append(depth_frame[tu[0]][tu[479]])
            else:
                distance_data.append(depth_frame[tu[0]-1][tu[1]-1])
            
        return distance_data

    def captureAndSaveImage(self):
        self.frames = self.getFrame()
        depth_frame = self.getDepthFrame(self.frames)
        color_frame = self.getRGBFrame()
        
        if not depth_frame or not color_frame:
            return

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        #gray_image = np.array(gray_image)
        #gray_image.convertTo(gray_image, CV_8UC3, 255.0); 
        path = '/home/pi/Desktop/test.png'
        #frame_normed = 255 * (color_image - color_image.min()) / (color_image.max() - color_image.min())
        #frame_normed = np.array(frame_normed, np.int)
        cv2.imwrite(path, 55*gray_image)
        
    def writeDepthDataToFile(self):
        schedule.run_pending()
        self.frames = self.getFrame()
        depth_frame = self.getDepthFrame(self.frames)
        depth_image = np.asanyarray(depth_frame.get_data())
        
        distances = []
        for x in depth_image:
            for y in x:
                distances.append(depth_frame.get_distance(x, y)[0])
        
        try:
            
            with open('/home/pi/Desktop/DepthData.npy', 'wb+') as f:
                np.save(f, distances, allow_pickle=True, fix_imports=True)
        except:
            raise

    def getCameraView(self):
            
            depth_image = np.load('/home/pi/Desktop/DepthData.npy')
            
            #dim = (160, 120)
 
            # resize image
            #resized = cv2.resize(depth_image, dim, interpolation = cv2.INTER_AREA)
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            gray_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
            
            #yield result
            # Get Distance when circle around things and the intel realsense camera is chosen
                
            (flag, encodedImage) = cv2.imencode(".jpg", gray_image)
            # Show images
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                    bytearray(encodedImage) + b'\r\n')
            #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('RealSense', images)
            #cv2.waitKey(1)

