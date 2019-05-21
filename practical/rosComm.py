import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import threading

class RosComm(threading.Thread):

    def callback(self, rgb_data, depth_data):
        print('got rgb-d image')
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
            self.latest_rgb = self.bridge.imgmsg_to_cv2(depth_data, "passthrough")
            #self.depth_array = np.array(self.depth_image, dtype=np.float32)
        except CvBridgeError:
            print('shiat')

    def run(self):

        #rospy.init_node('gqcnn')
        self.bridge = CvBridge()
        self.image_sub = message_filters.Subscriber("/camera/color/image_raw/", Image)
        self.depth_sub = message_filters.Subscriber("/camera/depth/image_rect_raw/", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 10, 0.5)
        self.cache = message_filters.Cache(self.ts, 10)
        self.ts.registerCallback(self.callback)
        print('shiat')
        #rospy.spin()
        print('shiat')
       
    def __init__(self, nodeName:str):
        self.latest_rgb = []
        self.latest_depth = []
        self.thread1 = threading.Thread(target = self.run)
        self.thread1.start()


    def getRGB(self):
        return self.latest_rgb

    def getDepth(self):
        return self.latest_depth