import threading
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
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
        self.latest_depth = self.bridge.imgmsg_to_cv2(depth_data, "passthrough")
        #self.depth_array = np.array(self.depth_image, dtype=np.float32)

    def threaded_synced_rgbd(self, *argv):
        self.image_sub = message_filters.Subscriber(argv[0], Image)
        self.depth_sub = message_filters.Subscriber(argv[1], Image)
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
        print(argv[0])
        self.subscriber_registry[argv[0]] = rospy.Subscriber(argv[0], argv[1], self.threaded_callback)
        rospy.spin()

    def subscribe(self, topic, msg_type):
        if self.subscriber_registry.get(topic):
            raise RuntimeError("Already registerd...")
        self.thread = threading.Thread(target = self.threaded_subscription, args = (topic, msg_type), name=topic)
        self.output_registry[topic] = []
        self.thread.start()
        
    def stop_subscription(self, topic):
        if self.output_registry.get(topic) is not None:
            self.subscriber_registry[topic].unregister()
            del self.output_registry[topic]

    def getLatestMessage(self, topic):
        return self.output_registry.get(topic) 
    
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