import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from threading import Thread

class RosComm(Thread):


    def __init__(self, nodeName:str, rgbTopic, depthTopic):
        Thread.__init__(self)
        self.latest_rgb = None
        self.latest_depth = None
        self.nodeName = nodeName
        self.rgbTopic = rgbTopic
        self.depthTopic = depthTopic
        

    def callback(self, rgb_data, depth_data):
        print('got rgb-d image')
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
            self.latest_rgb = self.bridge.imgmsg_to_cv2(depth_data, "passthrough")
            #self.depth_array = np.array(self.depth_image, dtype=np.float32)
        except CvBridgeError:
            print('shiat')

    def run(self):
        rospy.init_node(self.nodeName)
        self.bridge = CvBridge()
        self.image_sub = message_filters.Subscriber(self.rgbTopic, Image)
        self.depth_sub = message_filters.Subscriber(self.depthTopic, Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 10, 0.5)
        self.cache = message_filters.Cache(self.ts, 10)
        self.ts.registerCallback(self.callback)
        print('shiat')
        #rospy.spin()
        print('shiat')
       

    @property
    def rgb(self):
        return self.latest_rgb

    @property
    def depth(self):
        return self.latest_depth