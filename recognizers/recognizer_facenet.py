from recognizers.recognizer_base import FaceRecognizer
import tensorflow as tf 
import re
from tensorflow.python.platform import gfile
import os
import numpy as np 

class FaceRecognizerFaceNet(FaceRecognizer):
     '''
     FaceRecognizerFaceNet impelements FaceReconizer and perform the embedding process using facenet model
     
     Attributes:
         isloaded (bool) : True if the model is loaded and false otherwise
         sess (tensorflow session) : the local session to use when performing operations
         model_dir (string) : diractory to facenet model 
     '''
     NAME = 'recognizer facenet'
     
     def __init__(self):
         super(FaceRecognizer, self).__init__()
         self.is_loaded = False;
         self.sess =None
         self.input_map=None
         self.model_dir=os.path.dirname(os.path.realpath(__file__)) +'\\models\\facenetmodel\\20180402-114759'
         
     def __str__(self):
        return self.name()

     def name(self):
        return 'facenet recognizer'

     def load(self):
       '''
        load the wights of facenet model to prepare for use it also initialize the sess attribute 
        
        Returns:
            True if the model was loaded successfully 
       '''
       if  self.is_loaded:
            print('model is already loaded')
            return True
       
       self.sess = tf.Session()        
       with self.sess.as_default():  
            print('loading facenet model .... ')
            model_exp = os.path.expanduser(self.model_dir)
            if (os.path.isfile(model_exp)):
                with gfile.FastGFile(model_exp,'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, input_map= self.input_map, name='')
            else:
                meta_file, ckpt_file = self.get_model_filenames(model_exp)
                saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=self.input_map)
                saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
         
                 
            self.is_loaded= True
            return True


     def get_model_filenames(self,model_dir):
        '''
        helper function to get he names of model files in diractory 
        
        Args : 
            model_dir (string) : diractory of the model 
        Returns: 
           meta_file (list) : the names of meta files in diractory
           ckpt_file (list) : the names of cpkt files in diractory
        
        '''
        
        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files)==0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files)>1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
        meta_file = meta_files[0]
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
            return meta_file, ckpt_file
    
        meta_files = [s for s in files if '.ckpt' in s]
        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups())>=2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        return meta_file, ckpt_file
  
 


     def configure(self,model_dir,input_map):
       '''
       
       set the configuration of facenet model 
       
       Args:
         minsize (int) : the minimum size of detected faces
         threshold (list) : the float values to use in filtering the faces smaller values means smaller confidance
         facetor (float) : the scaling factor of the face 
         
      
       '''
       self.model_dir=model_dir
       self.input_map = input_map



     def extract_features(self, image,clean=True):
        '''
        
        embeds face image into vector of 512 value that represent the face using the facenet model 
        
        Args :
            image (list numpy array) : the input aligned face image or images of shape (n,3,160,160)
            clean (boolean) : if true it clean the loaded model 
            
        Returns : 
            parsed feature vector of shape (n,512)
        
        '''
        
        if not self.is_loaded :
            self.load()
        image = self.prewhiten(image)
        with self.sess.as_default():
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: image, phase_train_placeholder:False }
            emb =  self.sess.run(embeddings, feed_dict=feed_dict)
     
        if clean : 
            self.clean()
        return self.parse_output(emb)
    
    
    
    
     def parse_output(self,emb):
        '''
         helper function to parse ouput of the recognizer to fit in the pipline - can be ignored 
        '''
        return emb
        
     def clean(self):
        '''
         clean the loaded model 
         
        '''
        self.sess=None
        return True

     def prewhiten(self, x):
        '''
        
        helper function to normalize the input image by sutracting the mean and divide on the std so all values are between 0 and 1 
        
        Args :
            x (numpy array) : input array 
        
        Returns: 
            y (numpy array) : normalized input 
        
        '''
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y  