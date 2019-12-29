from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from recognizers.recognizer_base import FaceRecognizer
import numpy as np
import mxnet as mx
import os
import cv2
import sklearn.preprocessing

class FaceRecognizerArcFace(FaceRecognizer):
  '''
   FaceRecognizerArcFace impelements FaceReconizer and perform the embedding process using arcface  model
   
   Attributes: 
       prefix (string) : prefix path to arcface model 
       ctx (int) : the context of cpu/gpu usage set to mx.gpu() to use gpu and mx.cpu() is the default
       epoch (number) : the number of loaded model epoch default 0000 used in traing can be ignored 
       image_size (list) : the size of the input image best is default which is [112 112]
       model (mx model) : the arcface model 
       isloaded (boolean) : True if the model is loaded
       layer (string) : the name of the layer to use as feature vector default is 'fc1' which is the fist connected layer 
       
   
  '''
  NAME = 'recognizer arcface' 
  
  def __init__(self):
      super(FaceRecognizer, self).__init__()
      self.prefix = os.path.dirname(os.path.realpath(__file__)) +'\\models\\arcface\\model'
      self.clean=False
      self.ctx=mx.cpu()
      self.epoch=0000
      self.image_size=[112 ,112]
      self.layer='fc1'
      self.isloaded =False
      self.model=None
      

  def extract_features(self, aligned,clean =True):
      '''
        
        embeds face image into vector of 512 value that represent the face using the arface model 
        
        Args :
            aligned (list numpy array) : the input aligned face image or images of shape (n,3,160,160) 
            clean (boolean) : if true it clean the loaded model 
            
        Returns : 
            parsed feature vector of shape (n,512)
        
      '''
      emb =[]
      aligned = self.parse_input(aligned)
      if not self.isloaded:
          self.load()
      for image in aligned:
          input_blob = np.expand_dims(image, axis=0)
          data = mx.nd.array(input_blob)
          db = mx.io.DataBatch(data=(data,))
          self.model.forward(db, is_train=False)
          embedding = self.model.get_outputs()[0].asnumpy()
          embedding = sklearn.preprocessing.normalize(embedding).flatten()
          emb.append(embedding)
      if clean : 
           self.clean()
      return self.parse_output(emb)

  def load(self):
      '''
        load the wights of arcface model to prepare for use it also initialize the sess attribute 
        
        Returns:
            True if the model was loaded successfully 
      '''
      print('loading arcface model ...')
      sym, arg_params, aux_params = mx.model.load_checkpoint(self.prefix, self.epoch)
      all_layers = sym.get_internals()
      sym = all_layers[self.layer+'_output']  
      model = mx.mod.Module(symbol=sym, context=self.ctx, label_names = None)
      model.bind(data_shapes=[('data', (1, 3, self.image_size[0], self.image_size[1]))])
      model.set_params(arg_params, aux_params)
      self.model = model 
      self.isloaded =True
      return True

  def configure(self,prefix,epoch,layer,image_size,ctx= mx.gpu(0)):
      '''
       
       set the configuration of arcface model 
       
       Args:
         prefix (string) : prefix path to arcface model 
         ctx (int) : the context of cpu/gpu usage set to mx.gpu() to use gpu and mx.gpu() is the default
         epoch (number) : the number of loaded model epoch default 0000 used in traing can be ignored 
         image_size (list) : the size of the input image best is default which is [112 112]
         layer (string) : the name of the layer to use as feature vector default is 'fc1' which is the fist connected layer       
      
       '''
      self.prefix = prefix
      self.epoch = epoch
      self.layer = layer
      self.image_size=image_size
      self.ctx = ctx
      return True

  def name(self):
      return 'facenet recognizer'
  
  def parse_output(self,emb):
      '''
         helper function to parse ouput of the recognizer  to fit in the pipline - can be ignored 
      '''
      return np.asarray(emb)
        
  def clean(self):
      '''
         clean the loaded model 
         
      '''
      self.model=None
      return True
  def parse_input(self,images):
      '''
      helper function to parse input of the recognizer to fit in the pipline
      
      Args:
          images (list) : the unparesed input of images of faces 
    
     Returns 
         ret (numpy array): the pared input images of shape (n,3,size,size)
       
      '''
      ret = []
      for f in images  :
         
          tmp = cv2.resize(f,(112,112))
          tmp = np.transpose(tmp, (2,0,1))
          ret.append(tmp)

      ret = np.asarray(ret)

    
      return ret
      
