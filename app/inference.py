import tensorflow as tf      
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd + "/Mask_RCNN/samples/coco")                                                        
sys.path.append(cwd + "/Mask_RCNN") 
sys.path.append(cwd + "/Mask_RCNN/mrcnn")                                                          
                                                                              
import mrcnn.model as modellib                                                                                    
from asahi_coco import CocoConfig          


def get_config():
    config = CocoConfig()
    class InferenceConfig(config.__class__):                                                                        
        # Run detection on one image at a time                                                                      
        GPU_COUNT = 1                                                                                               
        IMAGES_PER_GPU = 1                                                                                          
        NUM_CLASSES = 2
        # BATCH_SIZE = 1

    config = InferenceConfig()
    return config


class MaskRCNN:
    def __init__(self):
        # DEVICE = "/cpu:0"                                                                                                 
        DEVICE = "/gpu:0"                                                                                               
        MODEL_DIR = "models"
        class_names = ['BG', 'upper_ellipse']
        self.graph = tf.get_default_graph()

        config = get_config()

        self.model = modellib.MaskRCNN(
            mode="inference", 
            model_dir=MODEL_DIR,                                            
            config=config
            )  

        self.model.model_dir = MODEL_DIR 

        weights_path = MODEL_DIR + '/mask_rcnn_coco_0032.h5'
        print("Loading weights ", weights_path)  
        self.model.load_weights(weights_path, by_name=True) 


    def predict(self, image):
        with self.graph.as_default():
            result = self.model.detect([image])[0]
                                                                                    
        return result


