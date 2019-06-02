
#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline
#%%
import matplotlib.pyplot as plt
from practical.webserver.sampleClient import predictMask, predictGQCNN
from practical.rosComm import RosComm
#%%
sys.path.append('.')
sys.path.append('./rai/rai/ry')
import rospy


#%%
#%%
rosco = RosComm()
#%% 
rospy.init_node('z')
#%%
rosco.subscribe_synced_rgbd('/camera/color/image_raw/', '/camera/depth/image_rect_raw/')

#%%
intr_rs = rosco.get_camera_intrinsics('/camera/color/camera_info')
#%%
img = rosco.rgb
d = rosco.temp_filtered_depth(20, blur ='bilateral', mode='median')
#%%
plt.imshow(img)
#%%
mask = predictMask(img, host="http://ralfi.nat.selfnet.de:5000",height=intr_rs['height'], width=intr_rs['width'], fx=intr_rs['fx'], fy=intr_rs['fy'], cx=intr_rs['cx'], cy=intr_rs['cy'])
#%%
plt.imshow(mask, cmap='gray')
#%%
grasp = sampleClient.predictFCGQCNN(img ,d ,mask, host='http://ralfi.nat.selfnet.de:5000',height=intr_rs['height'], width=intr_rs['width'], fx=intr_rs['fx'], fy=intr_rs['fy'], cx=intr_rs['cx'], cy=intr_rs['cy'])

#%%
res
#%% 
grasp = predictGQCNN(img, d, 'http://localhost:5000')

#%%
grasp

#%%
plt.imshow(mask['masks'][0])
#%%
