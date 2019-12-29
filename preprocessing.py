import  cv2
import numpy as np 

class preprocessing: 
 def process_image(self,image, rescale, recolor):
     if rescale['req']:
        image= self.rescale(image,rescale['width'], rescale['height'])
     if recolor['req']:
         image = self.rgb2gray(image)
     return image

 def rescale (self,image,width,height):
     image=  cv2.resize(image,(width,height))
     return image 
     
 def rgb2gray(self,image):
     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
     return image 
 
 def crop (self,image,boxes ):
     faces = []
     for box in boxes :
         x=int( round (box[0]))
         y=int( round (box[1]))
         w=int (round (box[2]) ) 
         h=int (round ( box[3]))
         cropped = image[y:h+y,x : w+x,:]
         faces.append(cropped)
         
     return faces
 
 def resize2square (self,image,x,y):
     resized= cv2.resize(image,(x,y),interpolation=cv2.INTER_AREA)
     return resized

 def preprocess_facenet(self, images):
     ret = np.zeros([len(images),160,160,3])
     for image in images  : 
        resized = self.resize2square(image,160,160)
        np.append(ret,resized)
     return ret          