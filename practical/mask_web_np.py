#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline
#%%
import matplotlib.pyplot as plt
from practical.webserver.sampleClient import predictMask, predictGQCNN
#%%
sys.path.append('.')
sys.path.append('./rai/rai/ry')
import libry as ry


#%%
cam = ry.Camera("testmask", "/camera/color/image_raw/", "/camera/depth/image_rect_raw", True)

#%%
img = cam.getRgb()
d = cam.getDepth()
#%%
plt.imshow(img)

#%%
res = predictMask(img, "http://localhost:5000")
#%%
res
#%% 
grasp = predictGQCNN(img, d, 'http://localhost:5000')

#%%
grasp


#%%
