import sys
sys.path.append('ry')
#import libry as ry
from datetime import datetime
import shutil
import os
import time
import rospy
import sensor_msgs
import numpy as np
import geometry_msgs
try:
    sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
from cv_bridge import CvBridge
import cv2

output_dir = "videos/" + str(datetime.now()) + "/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
nodename = "recordernode"
t = ["/camera/rgb/image_rect_color", "/cameras/left_hand_camera/image", "/cameras/right_hand_camera/image"]
dummy_depth_topic = "/camera/depth/image_rect_raw"

rospy.init_node("recordeBig", disable_signals=True)
recorders = []
codec = cv2.VideoWriter_fourcc(*'mp4v')
rate= rospy.Rate(30)
bridge = CvBridge()
img0 = np.zeros((1))
img1 = np.zeros((1))
img2 = np.zeros((1))

class ImageServ():

    def __init__(self, topic):
        self.img = None
        self.topic = topic
        self.sub = rospy.Subscriber(topic, sensor_msgs.msg.Image, self.callback)


    def callback(self, data):
        self.img = bridge.imgmsg_to_cv2(data, "bgr8")
        



s0 = ImageServ(t[0])
s1 = ImageServ(t[1])
s2 = ImageServ(t[2])


out_dir = output_dir + t[0].replace("/", "_") + ".mp4"
# chest
r = {
    "topic": t[0],
    "cam": s0, #rospy.Subscriber(t[0], sensor_msgs.msg.Image, callback)#"cam": ry.Camera(nodename, t, dummy_depth_topic, True),
    "out_dir" : out_dir,
    "writer" : cv2.VideoWriter(out_dir, codec, 30.0, (640, 480))
}
recorders.append(r)
out_dir = output_dir + t[1].replace("/", "_") + ".mp4"
#wristl
r = {
    "topic": t[1],
    "cam": s1, #rospy.Subscriber(t[1], sensor_msgs.msg.Image, callback) #ry.Camera(nodename, t, dummy_depth_topic, True),
    "out_dir" : out_dir,
    "writer" : cv2.VideoWriter(out_dir, codec, 30.0, (1280, 800))
}
recorders.append(r)
out_dir = output_dir + t[2].replace("/", "_") + ".mp4"
#wrist
r = {
    "topic": t[2],
    "cam": s2, #rospy.Subscriber(t[2], sensor_msgs.msg.Image, callback) #ry.Camera(nodename, t, dummy_depth_topic, True),
    "out_dir" : out_dir,
    "writer" : cv2.VideoWriter(out_dir, codec, 30.0, (640, 400))
}
recorders.append(r)

while True:
    try:
        rate.sleep()
        for r in recorders:
            f = r["cam"].img
            if f is not None:
                r["writer"].write(f)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        break


for r in recorders:
    r["writer"].release()

