import tensorflow as tf
from practical.tf_utils import loadRCNN
import numpy as np
import sys
import matplotlib.pyplot as plt
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
sys.path.append('../')
import time

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from practical.constants import retinaNetLabels

class RCNN():

    def __init__(self, path_name:str):
        ''' Create RCNN object for inference
        
        Args:
            path_name (str): path to a frozen tensorflow graph
        '''
        self.sess, self.x, self.classes, self.scores, self.boxes = loadRCNN(path_name)

    
    def predict(self, image:np.ndarray):
        ''' Predict classes, scores, bounding boxes.
        
        Args:
            image (np.ndarray): image array
        '''
        labels, scores, boxes = self.sess.run([self.classes, self.scores, self.boxes], {self.x: np.expand_dims(image, axis=0)})

        # TODO: decode and visualize
        return labels, scores, boxes




class RetinaNet:

    def __init__(self, file_name:str, backbone:str='resnet152'):
        '''Create model for inference with RetinaNet
        
        Args:
            file_name (str): path to hdf5 file of keras model.
            backbone (str): backbone name
        '''
        self.backbone = backbone
        self.model = models.load_model(file_name, backbone_name=backbone)

    
    def predict(self, image:np.ndarray, vis=False):
        ''' RetinaNet prediction on image
        
        Args:
            image (np.ndarray): image tensor !!! BGR !!!
        '''
        if vis:
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)

        if self.backbone == 'resnet152':
            image, scale = resize_image(image, min_side=600, max_side=800)
        else:
            image, scale = resize_image(image)
        
        image = np.stack((image, image[:, ::-1, :]), axis=0)
        boxes, scores, labels = self.model.predict_on_batch(image)
        boxes /= scale

        if vis:
            self.visualizePredictions(retinaNetLabels, draw, labels[0], scores[0], boxes[0])
        return labels, scores, boxes
    
    def visualizePredictions(self, idToLabels:list, image:np.ndarray, labels:np.ndarray, scores:np.ndarray, boxes:np.ndarray):
        ''' Visualize Predictions of RetinaNet
        
        Args:
            idToLabels (list): 
            image (np.ndarray): 
            labels (np.ndarray): 
            scores (np.ndarray): 
            boxes (np.ndarray): 
        '''
        for box, score, label in zip(boxes, scores, labels):
            if score < 0.3:
                break
            color = label_color(label)
            b = box.astype(int)
            draw_box(image, b, color=color)
            caption = "{} {:.3f}".format(idToLabels[label], score)
            draw_caption(image, b, caption)
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(image)
        
    