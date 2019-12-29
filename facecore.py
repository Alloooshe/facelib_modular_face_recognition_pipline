"""
 this file containes the class FaceCore. 
 
"""
import os 
os.path.dirname(os.path.realpath(__file__))
import sys 
sys.path.insert(0,os.path.dirname(os.path.realpath(__file__)))
from preprocessing import  preprocessing  
from detectors.detector_mtcnn import FaceDetectorMTCNN as det_mtcnn
from detectors.detector_dlib import FaceDetectorDlib as det_dlib
from detectors.detector_lffd import FaceDetectorLFFD as det_lffd
from aligners.align_2d import FaceAligner2D as algin_2d
from recognizers.recognizer_facenet import FaceRecognizerFaceNet as rec_facenet
from recognizers.recognizer_arcface import FaceRecognizerArcFace as rec_arcface
from person_face import Face
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import pickle
import cv2
from os.path import isfile,join
import numpy as np 
from tracking.sort import Sort 
from recognizers.facematch import FaceMatch


class FaceCore ():
    """
    FaceCore is the core library wich implements the differenet face recognition pipeline stages.
    
    FaceCore is the core library wich implements the differenet face recognition pipeline stages
    the library performs:  detection face point extraction  aligment recognition, each step can be
    done using one or more models: this models include : face detection and face features extraction
    models are (mtcnn,dlib model , pyramid box) aligment model is (2d aligment) face embedding models
    are (google facenet ).
    
    Attributes:
        detect_models (dictionary) : containes objects of detection models which helps save model initilization and loading time.
        recognize_models (dictionary) : containes objects of recognition models which helps save model initilization and loading time.
        align_models (dictionary) : containes objects of aligment methods.
        track_models (dictionary) : contains objects of tracking models.
    """
    def __init__(self):
        self.detect_models = {'mtcnn':None, 'dlib':None,'pyramidbox':None,'lffd':None}
        self.recognize_models = {'facenet':None,'arcface':None}
        self.align_models ={'2d':None}
        self.track_models = {'sort':None}
        self.pipline_model_options={'detect_model' : 'mtcnn', 'align_model':'2d' ,'recognize_model' : 'facenet','prediction_model':'knn','track_model' : 'sort'}
        self.pipline_rescale_options  ='auto'
        self.isloaded_data = False 
        self.isloaded_models= False
        self.data = None
        self.face_size_threshold=40
        self.preprocessor = preprocessing()
        self.matcher = FaceMatch()
 

    def load_models(self):
        """
        load the models specified in the options.
        
        load th pipeline models which are set using set options if not set it will load the default models.
        
        Args:
        
        Return:
            bool: True if all loading was successfully done.
            
        """
        det_model =  self.pipline_model_options['detect_model']
        if det_model == 'mtcnn' : 
                det = det_mtcnn()
                self.detect_models['mtcnn'] = det 
                self.detect_models['mtcnn'].load()
        if det_model == 'dlib' : 
                det = det_dlib()
                self.detect_models['dlib'] = det
                self.detect_models['dlib'].load()
        if det_model == 'lffd' : 
                det = det_lffd()
                self.detect_models['lffd'] = det
                self.detect_models['lffd'].load()
                
        rec_model =  self.pipline_model_options['recognize_model']
        if rec_model == 'facenet' : 
                rec = rec_facenet()
                self.recognize_models['facenet'] = rec
                self.recognize_models['facenet'] .load()
        if rec_model == 'arcface' : 
                rec = rec_arcface ()
                self.recognize_models['arcface'] = rec
                self.recognize_models['arcface'] .load()
                
            
        self.track_models['sort'] = Sort() 
        
        
        return True
    
    def set_model_options(self, model , options):
        """
        set the configuration of a model.
        
        if the model object is not create this function will create one and sotre it in the coressponding dictionary attribute
        note that the users should know what configuration each model accept and this fucntion uses configure function found 
        in each model class.
        
        Args:
            model (string) : the name of the module you wich to configure
            options (dictionary) : the options which the module accepts
        
        Return:
            bool: True if all configuration was successfully done.
        """
        if model =='mtcnn':
            minsize = options['minsize']
            threshold =options['threshold']
            factor= options['factor']
            if self.detect_models['mtcnn'] ==None :
                  det = det_mtcnn()
                  self.detect_models['mtcnn'] = det 
            self.detect_models['mtcnn'].configure(minsize,threshold,factor)
        return True
    
    def set_pipline_option(self,model_options,rescale_options='auto'):
        """
        set the models to use when performing pipline combined operations.
        
        set the models which the library should load and use when preforming one or more of the pipline options.
        
        Args:
            model_options (dictionary) : contains the names of detect,align,embedding models.
            rescale_options (string/array) : if set to auto this will let the library decide how to process input image or
                                             provide the width and hieght to rescale the image.
        
        Return:
            bool: True if all configuration was successfully done.
        """
        self.pipline_model_options=model_options 
        self.pipline_rescale_options= rescale_options
        self.read_data()
        return True
    
    def detect_faces(self,image, model=None, draw = True, clean =True,track_time=True):
     
        """
        detect the faces in an image.
        
        detect the faces bounding boxes and the face landmarks positions if the model supports that functionality
        
        Args:
            image (numpy array) : the target image.
            model (string) : the name of the model you wish to use .
            draw (bool) : if you wich to draw results or not .
            clean (bool) : if you wich to clear the models used in this process to save memory or keep it for faster response.
            track_time (bool) : if you wish to track the processing time.
        
        Return : 
            faces (numpy array) : the bounding boxes of the detected faces.
            points (numpy array) : the locations of the all faces landmarks may be None if the model doesn't support face landmarks.
            detect_time (float) : the time it tooks to performe detection.
        """
        if model ==None:
            model = self.pipline_model_options['detect_model']
        if model ==None: 
            model = 'mtcnn'
            
        detect_time= -1
        if self.detect_models[model] is None:     
            if model == 'mtcnn' : 
                det = det_mtcnn()
                self.detect_models['mtcnn'] = det 
            if model == 'dlib' : 
                det = det_dlib()
                self.detect_models['dlib'] = det
            if model == 'lffd' : 
                det = det_lffd()
                self.detect_models['lffd'] = det

        
        if track_time : 
            start = time.time()
        faces,points = self.detect_models[model].detect(image,clean)
       
        if track_time : 
            
            end = time.time()
            detect_time = end-start 
        if draw : 
           self.draw(image,faces,points) 
           
        if clean : 
            for idx in self.detect_models:
                    self.detect_models[idx]=None
        
        
      
        return faces,points,detect_time 
    
    
    def align (self,img,points,model=None,track_time = False,draw = False,raise_profie_flag =False):
        """
        align the image for better recognitions. 
        
        align using one of the allowed aligment models.
        
        Args:
            image (numpy array) : the target image.
            model (string) : the name of the model you wish to use .
            draw (bool) : if you wich to draw results or not .
            track_time (bool) : if you wish to track the processing time.
        
        Return:
            aligned (numpy array) : the aligned image.
            align_time (float) : the time it tooks to align the image.
        
            
        """
        if model ==None:
            model = self.pipline_model_options['align_model']
        if model ==None: 
            model = '2d'
        flag = False
        
      
        if model == '2d' : 
            self.align_models['2d']= algin_2d()
        start= 1 
        end=0
        if track_time : 
            start = time. time()
        
        aligned = self.align_models['2d'].align(img,points)
        eps =5
        if raise_profie_flag : 
            x= points.copy()
            x=x[0]
            e1 = x[0]
            e2=x [1]
            n = x[2]

            if abs (n[0] -e1[0] ) <eps  or abs(n[0] -e2[0]) <eps :
               flag =True 
               
        if track_time : 
            end = time.time()
        align_time = end-start 
        if draw : 
            for face in aligned:
                self.draw(face,None,None) 
        return aligned,align_time,flag
   
    def extract_featuers(self,img,model=None, track_time = False,clean=True ) :
        """
        extract face embedding from image.
        
        use a model to extract the presentation embedding of a face which is used in recognition and verfication 
        
        Args:
            image (numpy array) : the target face image.
            model (string) : the name of the model you wish to use .
            clean (bool) : if you wich to clear the models used in this process to save memory or keep it for faster response.
            track_time (bool) : if you wish to track the processing time.
        
        Return:
            emb (numpy array) : the embeddings of the input image.
            emb_time (float) : the time it tooks to extract embeddings.
        """
        if model ==None:
            model = self.pipline_model_options['recognize_model']
        if model ==None: 
            model = 'facenet'
        
        if self.recognize_models[model] is None:     
            if model == 'facenet' : 
                rec = rec_facenet()
                self.recognize_models['facenet'] = rec
            if model == 'arcface' : 
                rec = rec_arcface()
                self.recognize_models['arcface'] = rec
        start= 1 
        end=0
        if track_time : 
            start = time. time()
        emb = self.recognize_models[model].extract_features(img,clean)
      
        if track_time : 
            end = time.time()
        emb_time = end-start 
        
        if clean : 
            for idx in self.recognize_models:
                    self.recognize_models[idx]=None
                    
        return emb,emb_time
   
    def track (self,bbox, model = None):
       """
       tracks target throw a series of images.
       
       tracks the face throw images in ortder to avoid the need to preform all the pipline each time.
       
       Agrs:
           bbox (numpy array) : the bounding boxes of faces in the image which we went to track.
           model (string) : the model name you wish to use for tracking.
      
       Return:
           ret (numpy array) : the bounding boxes with tracking id.
           track_time (float) : the time it tooks to track.
           
       """
       if model ==None:
            model = self.pipline_model_options['track_model']
       if model ==None: 
            model = 'sort'
       
       if model == 'sort': 
            if self.track_models['sort'] == None : 
                self.track_models['sort'] = Sort()
            bbox = self.track_models['sort'].process_bbox(bbox)      
       start = time. time()
       ret = self.track_models[model].update(bbox)
       end = time. time()
       track_time = end-start 
       ret=ret[::-1]
       return ret,track_time

    def compare_face_size(self,bbox,threshold):
        """
        compare the size of a face box against a threshold.
        
        Args:
            bbox (numpy array) : the bounding boxes of faces in the image which we went to compare.
            threshold (float) : the theshold to compare agaist
       
        Return:
            bool (bool) : True if the face is smaller or False if it is bigger
                    
        """
        w=bbox[2]
        h=bbox[3]
        if w< threshold or h < threshold:
            return False
        return True 

    def to_xyhw (self,bbox):
       """
       covert box to differnet box representation.
       
       Args:
           bbox (numpy array) : the bounding boxes to change.
      
       Return:
           bbox (numpy array) : the bounding boxes with different representation.
       """
       for box in bbox : 
            box[2] = box[2]-box[0]
            box[3] = box[3]-box[1]
       return bbox

    def classify_faces(self,faces,points,threshold=None):
       """
       classify faces and coressponding landmarks to big and small ones.
       
       Args:
           faces (numpy array) : the bounding boxes of faces with tracking id.
           points (numpy array) : the faces landmarks.
           threshold (float) : the threshold to compare against.
           
       Return:
           small_res (list of dictionary) : the small faces with thier id,landmarks,bounding boxes.
           big_res (list of dictionary) : the big faces with thier id,landmarks,bounding boxes
            
           
       """
       if threshold == None : 
           threshold =self.face_size_threshold
       
       small_res= []
       big_res= []
       ret = {}
       faces = self.to_xyhw(faces)
       for i in range (0,faces.shape[0]) :  
            if self.compare_face_size(faces[i],threshold)==False:
                ret={'id' :faces[i][4],'points' : points[i],'bboxes': faces[i]}
                small_res.append(ret)
            else :
                ret={'id' :faces[i][4],'points' : points[i],'bboxes': faces[i]}
                big_res.append(ret)

        
       return small_res,big_res
    
    def save_face(self,face,file_name,save_dir='.//data'):
        """
        save a person face object as pickle object.
        
        Args:
            face (person face) : person face object to serilzie.
            file_name (string) : the pickle file name in which to save the object.
            save_dir (string) : save diractory. 
        
        Return :
            bool (bool) : True if saved successfully.
            
        
        """
        save_dir = os.path.join(save_dir, os.path.basename(file_name+'.pkl') ) 
        with open(save_dir, 'wb') as outfile:
            pickle.dump(face, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    def load_face(self,file_name,save_dir='.//data'):
        """
        load face object from pickle serilazation. 
        
        Args :
            file_name (string) : the file to retrive.
            save_dir (string) : the file diractory.
        
        Return :
            face (person face) : the deserialized object.
            
        """
        save_dir = os.path.join(save_dir, os.path.basename(file_name) ) 
        with open (save_dir,'rb') as outfile:
            face =  pickle.load(outfile)
        return face

    def pipline_image(self,image,preprocess=False,clean=False, draw =False):
      """
      perform the pipeline stages on an image.
      
      stages include : preprossing -> detection -> aligment -> embedding .
      
      Args: 
          image (numpy array) : the image to process.
          preprocess (bool) : wherther to preprocess of not.
          clean (bool) : whether to clean the model when finish.
          draw (bool) : whether to draw the detection and aligment results.
     
      Return :
         dictionary (dictionary) : the pipeline results which include the bounding boxes, points, embeddings , total time , image , success flag.
              
      """
      try:
        if preprocess:
            image = self.preprocess_image(image)
        total_time = 0 
        bboxes,points,time = self.detect_faces(image=image,model=self.pipline_model_options['detect_model'],draw=draw ,clean=clean,track_time=True)
        total_time+=time
        if len(points ) ==0  : 
             return {'result':False, 'image':image}
       
        aligned,time,flag = self.align(image,points,draw=False)
        aligned =self.filter_image(aligned)
       # self.draw(aligned[0],None)
       
        emb,time = self.extract_featuers(aligned, model =self.pipline_model_options ['recognize_model'] ,track_time=True, clean=clean)
        print(emb.shape)
        total_time+=time
        #print('total time = ',total_time)
        return {'result':True ,'bboxes':bboxes,'points':points,'embeddings':emb,'time':total_time,'image':image}

      except AssertionError:
          return {'result':False, 'image':image}
      
    def draw (self,img,bboxes,points=None,names=None):
        """
        helper function to draw results when asked to 
        
        Args:
            img (numpy array) : the image to draw on.
            boxes (numpy array) : the bounding boxes.
            points (numpy array) : the landmark points.
        
        Return : 
            img (numpy array) : the image with bboxes and points drwan on.
        
        """
        __,ax = plt.subplots(1)
        i=0
        if not bboxes is None:
            for bb in bboxes:
                if not bb is None:
                    cv2.rectangle(img, (int(bb[0]),int(bb[1]) ),(int(bb[0]+bb[2]) , int (bb[1] + bb[3]) )  , (255,255,0),2)   

                if not names is None : 
                    cv2.putText(img,'identity : '+names[i][0], (int(bb[0]),int(bb[1]-5) ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
                    i=i+1     
        if not points is None:
           for point in points :
               for pt in point:
                   x= int (round(pt[0]))
                   y= int (round(pt[1])) 
                   cv2.circle(img, (x, y), 2,(255,255,255)) 
                  
                    
        
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    
        return img

    def read_data (self, data_dir=''):
        """
        read the serlized pickle type data in a diractory
        
        Args:
            data_dir (string) : data diractory.
        
        Return:
            faces (list of person face) : list of person face objects which is the deserialized data in dir.
            
        
        """
        if data_dir=='':
            
            if self.pipline_model_options['recognize_model'] == 'arcface': 
               data_dir=os.path.dirname(os.path.realpath(__file__)) +'\\data\\arcface\\'
               
            else: 
               data_dir=os.path.dirname(os.path.realpath(__file__)) +'\\data\\facenet\\'
               
        print(data_dir)  
        files =[f for f in  os.listdir(data_dir) if isfile( join(data_dir,f) ) ]
        faces= []
        for file in files : 
                face = self.load_face(file,data_dir) 
                faces.append(face)
        face = np.squeeze(faces)
        self.data=faces
        self.isloaded_data=True
        self.matcher.data =self.data
        return faces

    def recognize (self,emb,prev, video_method ="vote" , model_name="kdtree" ):
        '''
        recognize the person name using the face embeddings provided 
        
        Args:
            emb (numpy array) : the face embeddings of size 512
            prev (dictionary) : dictionary that holds the previous information of this face which is used in video face recognition
            video_method (string) : the method to use in time series face recognition (currently only vote and feature fusion are supported)
            model_name (string) : the recognition model to use (knn kmeans kdtree softmax xgboost)
        
        Return :
            name (string) : the name of the person which the embeddings belong to
            dist (float) : the distance between the embeddings and the closest match 
            ttime (float) : the total time of the recognition process.
            
        
        '''
        s = time.time()
        if video_method =='vote': 
           name,dist = self.matcher.voting_recognition(emb,model_name,prev['prev_names'])
        elif video_method =='ff':
           name,dist= self.matcher.featurefusion_recognition(emb,model_name,prev['emb']) 
        elif video_method=='image' : 
           name,dist = self.matcher.match_image(emb,model_name)
        else :
            print('video method was not found ...')
        
        e=time.time()
        ttime = e-s
        return name,dist,ttime

    
    def process_labeled_data(self,  data_dir= os.path.dirname(os.path.realpath(__file__)) +'\\labeled_data\\',  out_dir = os.path.dirname(os.path.realpath(__file__))+'\\data\\'):
        """
        process labeld data into serialized embeddings.
        
        Args:
            data_dir (string) : the labled data diractory.
            out_dir (string) : the diractory to serialize the data.
        
        Return :
            bool (bool) : True if all processing was successful.
        """
        dirts =[f for f in  os.listdir(data_dir) if not isfile(join(data_dir,f))]
        #print(dirts)
        i = 0 
        for dirt in dirts:   
            folder_path =join(data_dir,dirt)
            
            files = [f for f in os.listdir( folder_path ) if  isfile(join(folder_path,f)) ]
            for file in files: 
                path = join( folder_path,file)
                print(path)
                image = cv2.imread(path)
                res = self.pipline_image(image,preprocess=False,clean=False,draw=False)
                if res['result'] :
                    face = Face(num=i,name= dirt, emb=res['embeddings'], points= res['points'], bboxes =res['bboxes'])
                    self.save_face(face,file_name=file+'-'+dirt,save_dir=out_dir)
            i+=1
        #print(files)
        return True
 
    
    def preprocess_image(self,image) : 
        """
        apply preprocessing options to image.
        
        Args:
            image (numpy array) : the target image.
        
        Return :
            image (numpy array) : the preprocessed image.
        
        """
       

        if self.pipline_rescale_options == 'auto':          
            image =  cv2.resize(image,(int(image.shape[1]/2),int(image.shape[0]/2)) )
            return image
        if self.pipline_rescale_options =="None":
            return image
        else:
            w=self.pipline_rescale_options[0]
            h=self.pipline_rescale_options[1]
            image =  self.preprocessor.rescale(image,int(w),int(h))
            return image

    def draw_on_frame(self,frame,res, frame_col= (0,0,0),font = cv2.FONT_HERSHEY_SIMPLEX,show=True):      
        """
        draw pipline results on frame.
        
        Args :
            frame (numpy array) : the target image or frame to draw on.
            res (dictionary) : a dictionary that contains the result of pipline appiled on the frame.
            frame_col (tuple)  : (x,y,z) gives the rgb componants of the color to use in drawing.
            font (open cv font) : the writing font.
        
        Return: 
            frame (numpy array) : the frame with results drawn on it.
        
        """
        for face in res :
            bb = face['bboxes']
          

            if not bb is None :
                #cv2.putText(frame,'det conv : '+str(round(bb[4],3) ), (int(bb[0]),int(bb[1]-15) ), font, 0.1, frame_col, 1, cv2.LINE_AA)
                cv2.rectangle(frame, (int(bb[0]),int(bb[1]) ),(int(bb[0]+bb[2]) , int (bb[1] + bb[3]) )  , frame_col,1)   
                cv2.putText(frame,'id: '+str(round(bb[4],3) ), (int(bb[0]),int(bb[1]+10) ), font, 0.3,frame_col, 1, cv2.LINE_AA)
                if not face['ttime'] is None  :
                   cv2.putText(frame,'time: '+ str(round(face['ttime'],2)) , (int(bb[0]),int(bb[1]+15) ), font, 0.5,frame_col, 1, cv2.LINE_AA)
                if not face['name'] is None:
                   cv2.putText(frame,'identity : '+face['name'], (int(bb[0]),int(bb[1]-5) ), font, 0.5, frame_col, 1, cv2.LINE_AA)
            points = face['points']
            if not points is None:
                i = 0                
                for pt in points:
                    
                    x= int (round(pt[0]))
                    y= int (round(pt[1])) 
                    cv2.circle(frame, (x, y), 1, frame_col) 
                  
        if show:
            cv2.imshow('View', frame)
        return frame

    def print_on_frame(self,frame,infos ,frame_col= (0,0,0),font = cv2.FONT_HERSHEY_SIMPLEX,show=True):
        """
        draw informations on target frame.
        
        Args :
            frame (numpy array) : the target image or frame to draw on.
            res (dictionary) : a dictionary that contains the result of pipline appiled on the frame.
            frame_col (tuple)  : (x,y,z) gives the rgb componants of the color to use in drawing.
            font (open cv font) : the writing font.
        
        Return: 
            frame (numpy array) : the frame with results drawn on it.        
        
        """
        i = 10 
        for info in infos:  
            cv2.putText(frame,info, (10 ,i ), font, 0.3, frame_col, 2, cv2.LINE_AA)
            i+=10
        if show:
            cv2.imshow('Webcam View', frame)
        return frame


    def get_old_new_id (self,prev_faces,big_res,keep_num = 10):
        '''
        helper function to distinguish between new and old faces that appears in frame 
        
        Args: 
            prev_faces (dictionary) : contains the informations of previoulsy tracked faces in video 
            big_res (dictionary) : informations of the faces in current frame recognized big enough to be considered 
            keep_num (int) : indicates how many frames you want a face to be tracked along 
        
        Return : 
            prev_faces (dictionary) : updated dictionary with faces that are tracked and currently big enough 
            new_id (list) : list of the ids of the newly recognized faces (not tracked before)
            old_id (list) : list of the old ids that was tracked for the maximum count of frame counts
            active_id (list) : list of ids that still can be tracked 
        
        '''
        if not len(prev_faces)  ==0 :
             prev_ids = np.asarray([x['id'] for x in prev_faces])
        else: 
             prev_ids= np.asarray([])
        big_ids=np.asarray([x['id'] for x in big_res])
        new_id=[]
        old_id=[]
        if not big_ids.size == 0: 
           new_id = [ x for x in big_ids if x not in prev_ids]  
           old_id = [ x for x in big_ids if x  in prev_ids]
        active_faces = [x for x in prev_faces if len(x['prev_names'])<keep_num ]
        active_id = [x['id'] for x in active_faces]
        active_id = [x for x in active_id if x in old_id]
           
        old_id =[x for x in old_id if x not in active_id] 
       
        prev_faces = [x for x in prev_faces if x['id'] in big_ids]
        return prev_faces,new_id,old_id,active_id

    def filter_image (self,image):
        '''
        helper function that filters face image in order to help the embedding process using opencv filtering funcitons
        
        Args:
            image (numpy array) : the image array you want to apply the filter to 
            
        Return : 
            image (numpy array) : the filtered image
        '''
        for img in image :
            lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
      
            img =np.array([img])
        return image

    def process_stream(self, frame,show_frame=False,prev_faces=[],keep_num=50,video_method= "vote"):
         '''
         general function that applies all pipline face recongition stages and keep track of faces used with any kind of streams
         
         Args: 
             frame (numpy array) : current frame to process
             show_frame (boolean) : weather to show the processed frame or not, useful when using webcame
             prev_faces (dictionary) : contains the informations of previoulsy tracked faces in video
             keep_num (int) : indicates how many frames you want a face to be tracked along 
             video_method (string) : recognize method to use (example : knn)
         
         Return : 
             frame (numpy array) : processed current frame with faces boxes, names , ids, facial points drawn on the frame diractly 
             prev_faces (dictionary) : the updated tracked face informations
             
         '''
         rec_model =  self.pipline_model_options['recognize_model']
         prd_model = self.pipline_model_options['prediction_model']
        # frame = self.core.preprocess_image(frame) 
        # frame = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))
         faces,points,detect_time=self.detect_faces(image=frame,draw=False,clean=False)
         #print('detect time ', detect_time)
         bwid,track_time = self.track(model = 'sort', bbox =faces)
         #frame = self.print_on_frame(frame,infos = [ 'detect time '+str(round(detect_time,2) ), 'tracking time '+str(round(track_time,2))],show=False)
         small_res,big_res = self.classify_faces(bwid,points,self.face_size_threshold)
        # print('big_res len' , len(big_res))
        # print('small_res len' , len(small_res))
         for face in small_res : 
              face['name']=None
              face['ttime'] = None 
         
         prev_faces,new_id,old_id,active_id = self.get_old_new_id(prev_faces,big_res,keep_num) 
        
        
         rec_time =0 
   
        
         for face in big_res  : 
            
             cur_id = face['id']
      
             if cur_id in new_id : 
            
                 aligned,align_time,flag = self.align(frame,[face['points']],draw=False,raise_profie_flag=True)
                 aligned = self.filter_image(aligned)
                 if flag :
                     #print('the flag is on ')
                     #self.draw(aligned[0],None,None)
                     continue
                 emb,emb_time = self.extract_featuers(aligned,model=rec_model, track_time = True,clean=False )
              #   print('emb time ' , emb_time)
                 face['emb'] = [emb]
                 name , dist ,rec_time = self.recognize ( emb , prev ={'prev_names':[] , 'emb': []}, video_method =video_method,model_name=prd_model) 
                 face['name'] = name
                 face['prev_names'] = [name]
                 face['ttime'] = align_time+emb_time+rec_time
                 prev_faces.append(face)
                
             elif cur_id in active_id : 
                 aligned,align_time ,flag= self.align(frame,[face['points']],draw=False, raise_profie_flag=True)
                 aligned = self.filter_image(aligned)
                 emb,emb_time = self.extract_featuers(aligned,model=rec_model, track_time = True,clean=False )
                 f = list(filter(lambda p: p['id'] == cur_id, prev_faces))[0]
                
                 if flag :
                     # print('the flag is on ')
                     # self.draw(aligned[0],None,None)
                      continue
                 f['emb'].append(emb)
               #  print ('cur id ',cur_id)
                 name , dist ,rec_time = self.recognize ( emb , prev =f, video_method =video_method,model_name=prd_model) 
                # print('name is ' ,name)
                 #f['prev_names'].append(name)
                 f['bboxes'] =face['bboxes']
                 f['points'] = face['points']
                 f['name'] = name
                 f['ttime'] = align_time+emb_time+rec_time
                 prev_faces=[i for i in prev_faces if not (i['id'] == cur_id)] 
                 prev_faces.append(f)
                 
             else : 
                 f = list(filter(lambda p: p['id'] == cur_id, prev_faces))[0]
                 f['bboxes'] =face['bboxes']
                 f['points'] = face['points']
                 prev_faces=[i for i in prev_faces if not (i['id'] == cur_id)] 
                 prev_faces.append(f)
             
    
         
         frame = self.draw_on_frame(frame,res =small_res , frame_col=(0,0,255),show=False)
         frame = self.draw_on_frame(frame,res =prev_faces ,frame_col=(255,0,255),show=show_frame)
         
       
         return frame,prev_faces
