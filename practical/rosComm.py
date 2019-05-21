import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class RosComm():
    
    def __init__(self, nodeName:str):
        #register TimeSynchronizer
        rospy.init_node(nodeName)
        self.bridge = CvBridge()
        self.image_sub = message_filters.Subscriber("/camera/color/image_raw/", Image)
        self.depth_sub = message_filters.Subscriber("/camera/depth/image_rect_raw/", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 100, 1.5)
        self.ts.registerCallback(self.callback)
        print('got rgb-d image')
        #self.cache = message_filters.Cache(self.ts, 10)
        rospy.spin()
       
    def callback(self, rgb_data, depth_data):
        print('got rgb-d image')
        try:
            self.image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_data, "passthrough")
            self.depth_array = np.array(self.depth_image, dtype=np.float32)
        except CvBridgeError:
            print('shiat')

    def getRGBDimage(self):
        print('do sth')
        #return self.ts.