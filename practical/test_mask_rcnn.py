#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline
#%%
import os
import argparse
from tqdm import tqdm
import numpy as np
import skimage.io as io
from copy import copy
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from autolab_core import YamlConfig
import sys
from sd_maskrcnn import utils
from sd_maskrcnn.config import MaskConfig
from sd_maskrcnn.dataset import ImageDataset
from sd_maskrcnn.coco_benchmark import coco_benchmark
from sd_maskrcnn.supplement_benchmark import s_benchmark

from mrcnn import model as modellib, utils as utilslib, visualize

#%%
sys.path.append('./robotics-course')
sys.path.append('./robotics-course/rai/rai/ry')
import libry as ry


#%%
cam = ry.Camera("testmask", "/camera/color/image_raw/", "/camera/depth/image_rect_raw", True)

#%%
img = cam.getRgb()

#%%
plt.imshow(img)

#%%
config = YamlConfig("practical/cfg/maskrcnn.yaml")
#%%
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with tf.Session(config=tf_config) as sess:
    set_session(sess)
    print("Benchmarking model.")

    # Create new directory for outputs
    output_dir = config['output_dir']
    utils.mkdir_if_missing(output_dir)

    # Save config in output directory
    config.save(os.path.join(output_dir, config['save_conf_name']))
    image_shape = config['model']['settings']['image_shape']
    config['model']['settings']['image_min_dim'] = min(image_shape)
    config['model']['settings']['image_max_dim'] = max(image_shape)
    config['model']['settings']['gpu_count'] = 1
    config['model']['settings']['images_per_gpu'] = 1
    inference_config = MaskConfig(config['model']['settings'])
    
    model_dir, _ = os.path.split(config['model']['path'])
    model = modellib.MaskRCNN(mode=config['model']['mode'], config=inference_config,
                              model_dir=model_dir)
    
    print(("Loading weights from ", config['model']['path']))
    model.load_weights(config['model']['path'], by_name=True)
    res = model.detect([img], verbose=1)

#%%
res


#%%
