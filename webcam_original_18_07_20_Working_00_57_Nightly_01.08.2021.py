######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
import pickle
#import threading
from threading import Thread
from threading import Timer
import importlib.util
import board
import neopixel
#import imutils

num_pixels = 14
pixel_pin = board.D10
ORDER = neopixel.RGB


pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.1, auto_write=True, pixel_order=ORDER)

pixels.fill((0, 0, 0))

global count1_global
global count2_global
global count3_global
global count4_global

count1_global=0
count2_global=0
count3_global=0
count4_global=0

delay1 = 3
delay2 = 3
delay3 = 3
delay4 = 3

cycle1=0
cycle2=0
cycle3=0
cycle4=0

#lock = threading.Lock()
with open('density.pickle', 'wb') as f:
    pickle.dump([count1_global, count2_global, count3_global, count4_global], f)
        
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Define VideoStream class to handle streaming of video from webcam in separate processing thread
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(1200,720),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(-1)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True
        
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Define traffic_density class to handle traffic weight in separate processing thread
class traffic_density:  
    def __init__(self):
        self._running = True

    def terminate(self):  
        self._running = False  

    def run(self):
        while self._running:
            
            global count1_global
            global count2_global
            global count3_global
            global count4_global
            
            a = []
            b = []
            c = []
            d = []
            
#            lock.acquire()
        
            for i in range(10):
                with open('density.pickle', 'rb') as f:
                    count1_global, count2_global, count3_global, count4_global = pickle.load(f)
                a.append(count1_global)
                b.append(count2_global)
                c.append(count3_global)
                d.append(count4_global)
                print("count1_global",count1_global)
                print("count2_global",count2_global)
                print("count3_global",count3_global)
                print("count4_global",count4_global)
                time.sleep(1)

            
            average1 = np.average(a)
            average2 = np.average(b)
            average3 = np.average(c)
            average4 = np.average(d)

            average1 = round (average1)
            average2 = round (average2)
            average3 = round (average3)
            average4 = round (average4)
            
            print("Average Density in Lane1:",average1)
            print("Average Density in Lane2:",average2)
            print("Average Density in Lane3:",average3)
            print("Average Density in Lane4:",average4)
            
#            lock.release()
 
#Create Class
traffic_density = traffic_density()
#Create Thread
traffic_density_thread = Thread(target=traffic_density.run) 
#Start Thread 
traffic_density_thread.start()
#traffic_density_thread.join()

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Define traffic_signal class to handle traffic counter in separate processing thread
class traffic_signal:  
    def __init__(self):
        self._running = True

    def terminate(self):  
        self._running = False  

    def run(self):
        while self._running:
            
#            lock.acquire()

            lane1_on = [10, 7, 4]
            lane2_on = [10, 7, 1]
            lane3_on = [10, 4, 1]
            lane4_on = [7, 4, 1]
        
            red = [255,0,0]
            green = [0,255,0]
            pixels.fill((0, 0, 0))
            
            print ("Traffic_Lane_1: ON")
            for k in lane1_on:
                pixels[k] = red
                pixels[3] = green
            
            time.sleep(delay1) #Density delay
            pixels.fill((0, 0, 0))
            
            print ("Traffic_Lane_2: ON")
            for k in lane2_on:
                pixels[k] = red
                pixels[6] = green
            
            time.sleep(delay1) #Density delay
            pixels.fill((0, 0, 0))
            
            print ("Traffic_Lane_3: ON")
            for k in lane3_on:
                pixels[k] = red
                pixels[9] = green
            
            time.sleep(delay1) #Density delay
            pixels.fill((0, 0, 0))
            
            print ("Traffic_Lane_4: ON")
            for k in lane4_on:
                pixels[k] = red
                pixels[12] = green
            
            time.sleep(delay1) #Density delay
            pixels.fill((0, 0, 0))            
            
#            lock.release()

#Create Class
traffic_signal = traffic_signal()
#Create Thread
traffic_signal_thread = Thread(target=traffic_signal.run) 
#Start Thread 
traffic_signal_thread.start()
#traffic_signal_thread.join()

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.2)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate
        
# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'     

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
#frame_rate_calc = 1
#freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)
       
def Lane1():
    #extract the region of image within the user rectangle             
    roi1 = frame1[100:710, 10:300]
    rgb1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2RGB)
    rgb_resized1 = cv2.resize(rgb1, (300, 300))
    input_data1 = np.expand_dims(rgb_resized1, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data1 = (np.float32(input_data1) - input_mean) / input_std
        
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data1)
    interpreter.invoke()

    # Retrieve detection results
    boxes1 = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes1 = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores1 = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    cycle1=0

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores1)):
        #if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
        #To find only Cars only: [https://stackoverflow.com/questions/60989775/how-can-i-use-tensorflow-lite-to-detect-specific-object-without-retraining]   
#         if classes1[i] != 2 or 5 or 7:
#             scores1[i]=0
        if ((scores1[i] > min_conf_threshold) and (scores1[i] <= 1.0)):
            ymin = int(max(1,(boxes1[i][0] * imH)))
            xmin = int(max(1,(boxes1[i][1] * imW)))
            ymax = int(min(imH,(boxes1[i][2] * imH)))
            xmax = int(min(imW,(boxes1[i][3] * imW)))
            
#             cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            cycle1 = cycle1 + 1
            count1_global = cycle1
    cv2.putText(frame,'COUNT1: {0:.2f}'.format(cycle1),(30,90),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    with open('density.pickle', 'wb') as f:
        pickle.dump([count1_global, count2_global, count3_global, count4_global], f)
        
def Lane2():
    #extract the region of image within the user rectangle             
    roi2 = frame1[100:710, 310:600]
    rgb2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB)
    rgb_resized2 = cv2.resize(rgb2, (300, 300))
    input_data2 = np.expand_dims(rgb_resized2, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data2 = (np.float32(input_data2) - input_mean) / input_std
        
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data2)
    interpreter.invoke()

    # Retrieve detection results
    boxes2 = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes2 = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores2 = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    cycle2=0

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores2)):
        #if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
        #To find only Cars only: [https://stackoverflow.com/questions/60989775/how-can-i-use-tensorflow-lite-to-detect-specific-object-without-retraining]   
#         if classes2[i] != 2 or 5 or 7:
#             scores2[i]=0
        if ((scores2[i] > min_conf_threshold) and (scores2[i] <= 1.0)):
            cycle2 = cycle2 + 1
            count2_global = cycle2
    cv2.putText(frame,'COUNT2: {0:.2f}'.format(cycle2),(330,90),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    with open('density.pickle', 'wb') as f:
        pickle.dump([count1_global, count2_global, count3_global, count4_global], f)
        
def Lane3():
    #extract the region of image within the user rectangle             
    roi3 = frame3[100:710, 610:900]
    img3 = cv2.cvtColor(roi3, cv2.COLOR_BGR2RGB)
    img_resized3 = cv2.resize(img3, (300, 300))
    input_data3 = np.expand_dims(img_resized3, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data3 = (np.float32(input_data3) - input_mean) / input_std
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data3)
    interpreter.invoke()

    # Retrieve detection results
    boxes3 = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes3 = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores3 = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    cycle3=0

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores3)):
        #if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
        #To find only Cars only: [https://stackoverflow.com/questions/60989775/how-can-i-use-tensorflow-lite-to-detect-specific-object-without-retraining]   
#         if classes3[i] != 2 or 5 or 7:
#             scores3[i]=0
        if ((scores3[i] > min_conf_threshold) and (scores3[i] <= 1.0)):
            cycle3 = cycle3 + 1
            count3_global = cycle3
    cv2.putText(frame,'COUNT3: {0:.2f}'.format(cycle3),(630,90),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    with open('density.pickle', 'wb') as f:
        pickle.dump([count1_global, count2_global, count3_global, count4_global], f)

def Lane4():
    #extract the region of image within the user rectangle             
    roi4 = frame4[100:710, 910:1200]
    img4 = cv2.cvtColor(roi4, cv2.COLOR_BGR2RGB)
    img_resized4 = cv2.resize(img4, (300, 300))
    input_data4 = np.expand_dims(img_resized4, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data4 = (np.float32(input_data4) - input_mean) / input_std
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data4)
    interpreter.invoke()

    # Retrieve detection results
    boxes4 = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes4 = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores4 = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    cycle4=0

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores4)):
        #if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
        #To find only Cars only: [https://stackoverflow.com/questions/60989775/how-can-i-use-tensorflow-lite-to-detect-specific-object-without-retraining]   
#         if classes4[i] != 2 or 5 or 7:
#             scores4[i]=0
        if ((scores4[i] > min_conf_threshold) and (scores4[i] <= 1.0)):
            cycle4 = cycle4 + 1
    cv2.putText(frame,'COUNT4: {0:.2f}'.format(cycle4),(930,90),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    with open('density.pickle', 'wb') as f:
        pickle.dump([count1_global, count2_global, count3_global, count4_global], f)

while True:
      
    #timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame10 = videostream.read()
    frame = frame10.copy()
    
    frame1 = frame.copy()
    frame2 = frame.copy()
    frame3 = frame.copy()
    frame4 = frame.copy()

    # rectangle for ROI >> Lane #1, Lane #2, Lane #3 & Lane #4
    cv2.rectangle(frame, (10, 100), (300, 710), (0, 0, 255), 4)
    cv2.rectangle(frame, (310, 100), (600, 710), (0, 255, 255), 4)
    cv2.rectangle(frame, (610, 100), (900, 710), (255, 255, 0), 4)
    cv2.rectangle(frame, (910, 100), (1200, 710), (255, 0, 0), 4)

    Lane1()
    Lane2()
    Lane3()
    Lane4()


    # Draw framerate in corner of frame
#    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
    
#     # Calculate framerate
#     t2 = cv2.getTickCount()
#     time1 = (t2-t1)/freq
#     frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
