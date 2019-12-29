from resources.align.helpers import FACIAL_LANDMARKS_68_IDXS
from resources.align.helpers import FACIAL_LANDMARKS_5_IDXS
import numpy as np
import cv2

class FaceAligner2D:
    '''
    class that implements FaceAligner and perform 2d aligment 
    
    Attributes:
        desiredLeftEye (tuple) :  cooridantion of the relative position of the left eye used to specify the between eyes distance default(0.32,0.32)
        desiredFaceWidth (int) : the desired width of the aligned face default : 160 
        desiredFaceHeight (int) : the desired height of the aligned facedefault same as width 
    '''
    def __init__(self ):
        self.desiredLeftEye = (0.32,0.32)
        self.desiredFaceWidth = 160
        self.desiredFaceHeight = None
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
    
    def name(self):
        '''
        returns the name of the model 
        
        '''
        return 'Align_2D'

    def configure(self,desiredLeftEye, desiredFaceWidth, desiredFaceHeight):
        '''
        set the desired configuration of the model 
        
        Args : 
            desiredLeftEye (tuple) :  cooridantion of the relative position of the left eye used to specify the between eyes distance default(0.32,0.32)
            desiredFaceWidth (int) : the desired width of the aligned face default : 160 
            desiredFaceHeight (int) : the desired height of the aligned facedefault same as width 
            
        '''
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
 
    def parse_output(self,faces):
        '''
        pares output to fit in the pipline expectations 
        
        Returns :
            faces (numpy array) : the aligned faces array 
            
        '''
        faces = np.asarray(faces)
     
        return faces
    
    def align(self, image, landmarks):
        '''
        preform the alignement process 
        
        Args: 
            image (numpy array) : the image of face to align
            landmarks (numpy array)  : containes the coordinations of the faces could be 68 landmarks in the case of dlib or 5 in the case of mtcnn
            
        Returns: 
            the parsed ouput numpy array of face images
        ''' 
        aligned_faces = []
        for shape in landmarks:
           
    
            if (len(shape)==68):
                (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
                (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
            else:
                (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
                (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]
                
        
            leftEyePts = shape[lStart:lEnd]
            rightEyePts = shape[rStart:rEnd]
            
            leftEyeCenter = leftEyePts.mean(axis=0).astype("float")
            rightEyeCenter = rightEyePts.mean(axis=0).astype("float")
            
            dY = rightEyeCenter[1] - leftEyeCenter[1]
            dX = rightEyeCenter[0] - leftEyeCenter[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180
            desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
    
    		# determine the scale of the new resulting image by taking
    		# the ratio of the distance between eyes in the *current*
    		# image to the ratio of distance between eyes in the
    		# *desired* image
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
            desiredDist *= self.desiredFaceWidth
            scale = desiredDist / dist
    
    		# compute center (x, y)-coordinates (i.e., the median point)
    		# between the two eyes in the input image
            eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2,
    			(leftEyeCenter[1] + rightEyeCenter[1]) / 2)
    
    		# grab the rotation matrix for rotating and scaling the face
            M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    
    		# update the translation component of the matrix
            tX = self.desiredFaceWidth * 0.5
            tY = self.desiredFaceHeight * self.desiredLeftEye[1]
            M[0, 2] += (tX - eyesCenter[0])
            M[1, 2] += (tY - eyesCenter[1])
    
    		# apply the affine transformation
            (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
            output = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC)
            aligned_faces.append(output)
        return self.parse_output(aligned_faces)
