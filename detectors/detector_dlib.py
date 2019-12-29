'''
face detection using dlib with hog features
'''
import os
import dlib
import numpy as np
from .dlib.confs.conf import DeepFaceConfs
from .detector_base import FaceDetector


class FaceDetectorDlib(FaceDetector):
    """
    reference : https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    
    uses dlib library and feature based algorithm to get the bounding boxes and face feature points in an image
    
    
    
    """
    NAME = 'detector_dlib'
    predictor_path = None
   
    def __init__(self):
        super(FaceDetectorDlib, self).__init__()
        self.detector = None
        self.predictor = None
        self.upsample_scale = None
        self.is_loaded = False

    def name(self):
        '''
        returns the name of the model 
        
        '''
        return FaceDetectorDlib.NAME

    def detect(self, npimg,clean=False):      
        '''
         perform dlib face detection in image
         
         Args: 
             npimg (numpy Array) : the image to detect
             clean (boolean) : flag to indicate if you wish to clean the loaded model and configuration or not 
        
         Returns : 
             boxes (numpy array) : the bounding boxes of the faces in the image of shape (n,x,y,w,h,convidance)
             points (numpy array) : the coordinations of the face feature points of shape (n,2,68)
         '''
        if not self.is_loaded:
            self.load()
            self.is_loaded=True
        dets, scores, idx = self.detector.run(npimg, self.upsample_scale, -1)
     
        points=[]
        boxes=[]
        for det, score in zip(dets, scores):
            if score < DeepFaceConfs.get()['detector']['dlib']['score_th']:
                continue

            x = max(det.left(), 0)
            y = max(det.top(), 0)
            w = min(det.right() - det.left(), npimg.shape[1] - x)
            h = min(det.bottom() - det.top(), npimg.shape[0] - y)

            if w <= 1 or h <= 1:
                continue

            bbox = np.array([x, y, w, h, score])
         
            # find landmark
            face_landmark = self.detect_landmark(npimg, det)
            
            boxes.append(bbox)
            points.append(face_landmark)
        
    
        if clean :
             self.clean()
        
        return self.parse_output(boxes,points)

    def detect_landmark(self, npimg, det):
        '''
        get the face landmarks in an image 
        
        Args:
            npimg (numpy array) : the target image 
            det (dlib model) : the dlib detector model  to use 
            
        Returns:
            coords (list) : list of coordinations for face landmarks 
        '''
        shape = self.predictor(npimg, det)
        coords = np.zeros((68, 2), dtype=np.int)

        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    
    def parse_output(self,boxes,points):
        '''
        helper function to parse ouput of the detector to fit in the pipline
        
        Args:
           boxes (numpy array) : the faces bounding boxes of the image 
           points (numpy array) : the coordinations of the face feature points
           
        Returns:
            boxes (numpy array) : the faces bounding boxes of the image  in shape (n,x,y,w,h,confidance)
            points (numpy array) : the coordinations of the face feature points in shape (n,2,68)
        '''
        if len(points) >1:
            points= np.squeeze(points)
            boxes=np.squeeze(boxes)
        else:
            points = np.asarray(points)
            boxes= np.asarray(boxes)
        return boxes,points
    
    def load(self):
        '''
        load the dlib model to prepare for use
        
        Returns:
            True if the model was loaded successfully and False elsewise
        '''
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            DeepFaceConfs.get()['detector']['dlib']['landmark_detector']
        )
        self.predictor = dlib.shape_predictor(predictor_path)
        self.upsample_scale = DeepFaceConfs.get()['detector']['dlib']['scale']
        return True 
        
    def configure(self):
        return 0 
    
    def clean (self):
        '''
         clean the loaded model 
         
        '''
        self.predictor=None
        self.detector=None
        self.upsample_scale=None
        self.is_loaded =False
