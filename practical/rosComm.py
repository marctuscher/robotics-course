import threading
import time
import rospy
import message_filters
import sensor_msgs
from cv_bridge import CvBridge, CvBridgeError
import sys
import numpy as np
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2

class RosComm:

    def __init__(self):
        self.bridge = CvBridge()
        self.output_registry = {}
        self.subscriber_registry = {}
        # dont do this
        self.latest_rgb = None
        self.latest_depth = None
        
    def threaded_synced_rgbd_cb(self, rgb_data, depth_data):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")

        self.d = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")
        self.latest_depth = np.array(self.d, dtype=np.float32) * 0.001
        #self.latest_depth  = cv2.normalize(self.depth_array, self.depth_array, 0, 1, cv2.NORM_MINMAX)
        #self.depth_array = np.array(self.depth_image, dtype=np.float32)

    def threaded_synced_rgbd(self, *argv):
        self.image_sub = message_filters.Subscriber(argv[0], sensor_msgs.msg.Image)
        self.depth_sub = message_filters.Subscriber(argv[1], sensor_msgs.msg.Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], queue_size=10, slop=0.5)
        #self.cache = message_filters.Cache(self.ts, 10)
        self.ts.registerCallback(self.threaded_synced_rgbd_cb)
        rospy.spin()

    def subscribe_synced_rgbd(self,rgbTopic, depthTopic):
        self.thread = threading.Thread(target = self.threaded_synced_rgbd, args = (rgbTopic, depthTopic), name = rgbTopic + depthTopic)
        self.thread.start()
       
    def threaded_callback(self, data):
        # redirect output
        self.output_registry[threading.currentThread().getName()] = (data)

    def threaded_subscription(self, *argv):
        #print(argv[0])
        self.subscriber_registry[argv[0]] = rospy.Subscriber(argv[0], argv[1], self.threaded_callback)
        rospy.spin()

    def subscribe(self, topic, msg_type):
        if self.subscriber_registry.get(topic):
            raise RuntimeError("Already registerd...")
        self.thread = threading.Thread(target = self.threaded_subscription, args = (topic, msg_type), name=topic)
        self.output_registry[topic] = []
        self.thread.start()
        
    def stop_subscription(self, topic):
        if self.subscriber_registry[topic] is not None:
            self.subscriber_registry[topic].unregister()
            del self.output_registry[topic]
            del self.subscriber_registry[topic]

    def get_latest_message(self, topic):
        return self.output_registry.get(topic) 

    def get_camera_intrinsics(self, topic):
        self.subscribe(topic, sensor_msgs.msg.CameraInfo)
        time.sleep(.1)
        msg = self.get_latest_message(topic)
        self.stop_subscription(topic)
        # build intr dict
        intr = {}
        intr['frame_id'] = msg.header.frame_id
        intr['K'] = msg.K
        intr['P'] = msg.P
        intr['height'] = msg.height
        intr['width'] = msg.width
        intr['fx'] = msg.K[0]
        intr['fy'] = msg.K[4]
        intr['cx'] = msg.K[2]
        intr['cy'] = msg.K[5]
        return intr

    def temp_filtered_depth(self, numImages=10, blur = 'bilateral', mode= 'median', filter=True):
        arr = np.zeros([numImages,480,640])
        #blur =  [gaussian', 'bilateral', 'median']
        for i in range(numImages):
            #time.sleep(0.001 * 33.4) # sleep until new photo arrives
            if filter and blur == 'bilateral':
                arr[i,:,:] = cv2.bilateralFilter(self.latest_depth, 9,75,75)
            elif filter and blur == 'gaussian':
                arr[i,:,:] = cv2.GaussianBlur(self.latest_depth,(3,3), 0)
            elif filter and blur == 'median':
                arr[i,:,:] = cv2.medianBlur(self.latest_depth,5)
            else:
                arr[i,:,:] = self.latest_depth
        if mode == 'mean':
            fil_depth = np.nanmean(arr, axis=0, keepdims=True)
        elif mode == 'median':
            fil_depth = np.nanmedian(arr, axis=0, keepdims=True)
        else:
            fil_depth = np.nanmean(arr, axis=0, keepdims=True)

        return (fil_depth[0,:,:]).astype('float32')
    
    @property
    def output_reg(self):
        return self.output_registry

    @property
    def subscriber_reg(self):
        return self.subscriber_registry

    @property
    def rgb(self):
        return self.latest_rgb

    @property
    def depth(self):
        return self.latest_depth