'''
face detection using lffd
'''

from .detector_base import FaceDetector
import mxnet as mx
import sys
from .lffd import predict
import os 
import numpy as np

class FaceDetectorLFFD (FaceDetector):
    NAME = 'detector_lffd'
     
    def __init__(self):
         super(FaceDetectorLFFD, self).__init__()
         self.version = 'v1'
         self.ctx = mx.cpu()
         self.isloaded =False
         self.ctx = mx.cpu()
         self.face_predictor= None
         
    def name(self) :
        '''
        returns the type of the model.
        
        returns the type of the detection model instance that was used to call this function.
        
        Returns: 
            (string) : type of the detection model.
        
        '''
        return self.Name 
    
    def load(self):
        '''
        load lffd model.
        
        Returns:
            True if the model was loaded successfully
        '''
        if self.version == 'v1':
            from .lffd.config_farm import configuration_10_560_25L_8scales_v1 as cfg
            symbol_file_path = os.path.dirname(os.path.realpath(__file__)) +'.\\lffd\\symbol_farm\\symbol_10_560_25L_8scales_v1_deploy.json'
            model_file_path = os.path.dirname(os.path.realpath(__file__)) +'.\\lffd\\saved_model\\configuration_10_560_25L_8scales_v1\\train_10_560_25L_8scales_v1_iter_1400000.params'
        elif self.version == 'v2':
            from .lffd.config_farm  import configuration_10_320_20L_5scales_v2 as cfg
            symbol_file_path = './lffd/symbol_farm/symbol_10_320_20L_5scales_v2_deploy.json'
            model_file_path = './lffd/saved_model/configuration_10_320_20L_5scales_v2/train_10_320_20L_5scales_v2_iter_1800000.params'
     
        self.face_predictor = predict.Predict(mxnet=mx,
                                         symbol_file_path=symbol_file_path,
                                         model_file_path=model_file_path,
                                         ctx=self.ctx,
                                         receptive_field_list=cfg.param_receptive_field_list,
                                         receptive_field_stride=cfg.param_receptive_field_stride,
                                         bbox_small_list=cfg.param_bbox_small_list,
                                         bbox_large_list=cfg.param_bbox_large_list,
                                         receptive_field_center_start=cfg.param_receptive_field_center_start,
                                         num_output_scales=cfg.param_num_output_scales)
        self.isloaded=True
        return True
    
    def detect(self,img,clean =False):
        '''
         perform mtcnn face detection on image.
         
         Args: 
             img (numpy Array) : the image to detect
             clean (boolean) : flag to indicate if you wish to clean the loaded model and configuration or not 
        
         Returns : 
             boxes (numpy array) : the bounding boxes of the faces in the image of shape (n,5), output would be like [[x,y,w,h,convidance],...,[...]]
         '''
         
        if not self.isloaded  :
            print('loading lffd model...')
            self.load()
            print('lffd model was loaded successfully')
        bboxes, _ = self.face_predictor.predict(img, resize_scale=1, score_threshold=0.6, top_k=10, \
                                                        NMS_threshold=0.4, NMS_flag=True, skip_scale_branch_list=[])

        return self.parse_output(bboxes),[]
    
    def configure(self,use_gpu=False,version='v1'):
        '''
       
        set the configuration of face detection mtcnn model.
       
       
       Args:
         use_gpu (bool) : flag to set configuration to use gpu, when set true the default gpu unit is gpu number 0.
         version (string): the version of lffd to use, the possible values are 'v1' and 'v2'
         
       '''
   
        if args.use_gpu:
            self.ctx = mx.gpu(0)
        else:
            self.ctx = mx.cpu()
        
        self.version=version
   
        
        return True 
    
    def parse_output(self,bboxes):
        '''
        helper function to ouput bounding boxes and feature points in the proper shape.
        
        function that normlize the shape of bounding boxes array and feature points to be in the same shape.
        
        Args:
           boxes (numpy array) : the faces bounding boxes of the image 
           
        Returns:
            boxes (numpy array) : the faces bounding boxes of the image of shaspe (n,5), output would be like [[x,y,w,h,convidance],...,[...]]
        '''
        bboxes =np.array(bboxes)
   
        for box in bboxes:
            box[2]=box[2]-box[0]
            box[3]=box[3]-box[1]
        
        return bboxes 
    
    def clean(self):
        '''
         clean the loaded model and configuration 
         
        '''
        self.isloaded = False
        self.face_predictor =None
        return True
         
