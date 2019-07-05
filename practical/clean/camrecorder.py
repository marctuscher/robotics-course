import sys
sys.path.append('ry')
import libry as ry
import cv2
from datetime import datetime
import shutil
import os
import time
import rospy


output_dir = "videos/" + str(datetime.now()) + "/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
nodename = "recordernode"
topics = ["/camera/rgb/image_rect_color", "/cameras/left_hand_camera/image", "/cameras/right_hand_camera/image"]
dummy_depth_topic = "/camera/depth/image_rect_raw"


rospy.init_node("recordeBig", disable_signals=True)
recorders = []
codec = cv2.VideoWriter_fourcc(*'mp4v')
rate= rospy.Rate(30)

for i, t in enumerate(topics):
    out_dir = output_dir + t.replace("/", "_") + ".mp4"
    if i == 0:
        r = {
            "topic": t,
            "cam": ry.Camera(nodename, t, dummy_depth_topic, True),
            "out_dir" : out_dir,
            "writer" : cv2.VideoWriter(out_dir, codec, 20.0, (640, 480))
        }
    else:
        r = {
            "topic": t,
            "cam": ry.Camera(nodename, t, dummy_depth_topic, True),
            "out_dir" : out_dir,
            "writer" : cv2.VideoWriter(out_dir, codec, 20.0, (640, 400))
        }
    recorders.append(r)

while True:
    try:
        rate.sleep()
        for r in recorders:
            f = r['cam'].getRgb()
            cv2.imshow(r["topic"], f )
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            if r["topic"] == "/camera/rgb/image_rect_color":
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            else:
                f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            r["writer"].write(f)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        break


for r in recorders:
    r["writer"].release()

