from autolab_core import YamlConfig
from sd_maskrcnn import utils
from sd_maskrcnn.config import MaskConfig
import os
import tensorflow as tf
from mrcnn import model as modellib
import numpy as np
from keras.backend.tensorflow_backend import set_session, clear_session

class MaskLoader():

    def __init__(self, cfgFile="practical/cfg/maskrcnn.yaml"):
        config = YamlConfig(cfgFile)
        print("Benchmarking model.")

        # Create new directory for outputs
        output_dir = config['output_dir']
        utils.mkdir_if_missing(output_dir)

        # Save config in output directory
        image_shape = config['model']['settings']['image_shape']
        config['model']['settings']['image_min_dim'] = min(image_shape)
        config['model']['settings']['image_max_dim'] = max(image_shape)
        config['model']['settings']['gpu_count'] = 1
        config['model']['settings']['images_per_gpu'] = 1
        inference_config = MaskConfig(config['model']['settings'])
        
        model_dir, _ = os.path.split(config['model']['path'])
        self.model = modellib.MaskRCNN(mode=config['model']['mode'], config=inference_config,
                                model_dir=model_dir)
    
        print(("Loading weights from ", config['model']['path']))
        self.model.load_weights(config['model']['path'], by_name=True)
        self.graph = tf.get_default_graph()
    def predict(self, depth_img):
        with self.graph.as_default():
            img = np.array([depth_img, depth_img, depth_img])
            img = np.transpose(img, axes=[1, 2, 0])
            res = self.model.detect([img], verbose=0)[0]
            pred = {
                'rois': res['rois'].tolist(),
                'class_ids': res['class_ids'].tolist(),
                'scores': res['scores'].tolist(),
                'masks': res['masks'].tolist()
            }
        return pred