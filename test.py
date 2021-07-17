import numpy as np
import cv2
import time
from pylepton.Lepton3 import Lepton3
import threading
import json
import time
import RPi.GPIO as GPIO           # import RPi.GPIO module  

class ThermalCamera():
    def __init__(self):
        #self.data = self.cameraCapture()
        self.val = 1
        self.thermal_data = []
    def thermalData(self):
        timeout = time.time() + 15   # 1 minutes from now
        a = None
        while True:
            with Lepton3() as l:
                a,b = l.capture()
            if time.time() > timeout:
                break
        return a
    def cameraReset(self):
        GPIO.setmode(GPIO.BOARD)            # choose BCM or BOARD  
        GPIO.setup(40, GPIO.OUT) # set a port/pin as an output  
        GPIO.output(40, 0)       # set an output port/pin value to 0/LOW/Fa
        time.sleep(2)
        GPIO.output(40,1)
    def cameraCapture(self):
        while True:
            with Lepton3() as l:
              a,b = l.capture()
            
              minimum = a[a>0].min()
              minimum_temp = (minimum * 0.01) - 273.15
              maximum = a[a>0].max()
              maximum_temp = (maximum * 0.01) - 273.15
              if maximum_temp > 300:
                  continue
            #font = cv2.FONT_HERSHEY_SIMPLEX
            # Create a black image
            cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX) # extend contrast
            np.right_shift(a, 8, a) # fit data into 8 bits
            #cv2.putText(a,str('min {:.2f} max {:.2f}'.format(minimum_temp, maximum_temp)),(5, 20), font, 0.34,(255,0,0),1,cv2.LINE_4)
            #a = cv2.applyColorMap(np.uint8(a), cv2.COLORMAP_PLASMA)
            # Yeild Instead
            #if a is None:
            #    continue
            #a = cv2.resizeWindow('output', 600,600)
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", a)

            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')
            
        #cv2.namedWindow('output',cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('output', 600,600)
        #cv2.imshow("output", np.uint8(a)) # write it!


