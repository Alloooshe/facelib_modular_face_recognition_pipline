'''
face detection using mtcnn
'''

from detectors.detector_base import FaceDetector
import tensorflow as tf
from .mtcnn import mtcnn_detect_face as mtcnn_detector
import numpy as np 

class FaceDetectorMTCNN(FaceDetector):
     '''
     FaceDetcorMtCNN uses mtcnn to detect faces in images implement FaceDetector
     
     Attributes:
         isloaded (bool) : True if the model is loaded and false otherwise
         pnet (tensors) : the tensor that repesent the  pnet architicture 
         rnet (tensors) : the tensor that repesent the  pnet architicture 
         onet (tensors) : the tensor that repesent the  pnet architicture 
         minsize (int) : the minimum size of detected faces
         threshold (list) : the float values to use in filtering the faces smaller values means smaller confidance
         facetor (float) : the scaling factor of the face 
     
     '''
     NAME = 'detector_mtcnn'
     
     def __init__(self):
        super(FaceDetectorMTCNN, self).__init__()
        self.is_loaded = False
        self.pnet=None
        self.rnet=None
        self.onet=None
        self.minsize=20
        self.threshold=[ 0.6, 0.7, 0.7 ]
        self.factor= 0.709
        
        
     def name(self):
        '''
        returns the type of the model.
        
        returns the type of the detection model instance that was used to call this function.
        
        Returns: 
            (string) : type of the detection model.
        
        '''
        return FaceDetectorMTCNN.NAME
    
     def detect(self,img,clean =False):
         '''
         perform mtcnn face detection on image.
         
         Args: 
             img (numpy Array) : the image to detect
             clean (boolean) : flag to indicate if you wish to clean the loaded model and configuration or not 
        
         Returns : 
             boxes (numpy array) : the bounding boxes of the faces in the image of shape (n,5), output would be like [[x,y,w,h,convidance],...,[...]]
             points (numpy array) : the coordinations of the face feature points of shape (n,2,5)
         '''
   
         if not self.is_loaded:
            self.load()
        
         boxes, points = mtcnn_detector.detect_face(img,self.minsize,self.pnet,  self.rnet,  self.onet,self.threshold,  self.factor )
         if clean :
             self.clean()
         return  self.parse_output(boxes,points)
   
     def load (self):
        '''
        load the wights of mtcnn model.
        
        Returns:
            True if the model was loaded successfully. 
        '''
        print('loading mtcnn detection model ....')
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                pnet, rnet, onet = mtcnn_detector.create_mtcnn(sess, None)
                self.pnet=pnet
                self.rnet=rnet
                self.onet=onet
                self.is_loaded= True;
               
        print('mtcnn detection model was loaded successfully')
        return True
     
     def configure(self,minsize,threshold,factor):
       '''
       
       set the configuration of face detection mtcnn model.
       
       
       Args:
         minsize (int) : the minimum size of detected faces
         threshold (list) : the float values to use in filtering the faces smaller values means smaller confidance
         facetor (float) : the scaling factor of the face 
         
      
       '''
       self.minsize =  minsize 
       self.threshold =  threshold 
       self.factor  =factor 
       return True
   
     def parse_output(self,boxes,points):
        '''
        helper function to ouput bounding boxes and feature points in the proper shape.
        
        function that normlize the shape of bounding boxes array and feature points to be in the same shape.
        
        Args:
           boxes (numpy array) : the faces bounding boxes of the image 
           points (numpy array) : the coordinations of the face feature points
           
        Returns:
            boxes (numpy array) : the faces bounding boxes of the image of shaspe (n,5), output would be like [[x,y,w,h,convidance],...,[...]]
            points (numpy array) : the coordinations of the face feature points in shape (n,2,5)
        '''

        for box in boxes:
            box[2]=box[2]-box[0]
            box[3]=box[3]-box[1]
       
        points = np.transpose(points)
        pts=[]
        for point in points:
            pt = []
            for i in range( 0,5) : 
                x= point[i]
                y=point[i+5]
            
                pt.append(np.squeeze([x,y]) )
           
            pt= np.asarray(pt)
            pt = np.squeeze(pt)
            pts.append(pt)
        
        if len(pts) > 1 :
            pts= np.squeeze(pts)
        else:
            pts = np.asarray(pts)
       
        return boxes,pts

        
     def clean(self):
         '''
         clean the loaded model and configuration
    
         '''
        
         self.pnet=None
         self.rnet=None
         self.onet=None
         self.is_loaded=False
        