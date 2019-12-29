import cv2
import sys
import os 
from os.path import isfile,join
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
sys.path.insert(0,"current/dir")
import matplotlib.pyplot as plt
from facecore import FaceCore
core= FaceCore()
import numpy as np 


def validate_lfw(eps=1 , pairs_path  = "",data_dir="" ) : 
    
    with open (pairs_path,'rb') as file:
            FAR= 0
            FRR =0
            bool_result=[]
            j=0
            lines = file.readlines()
            for line in lines:   
                j=j+1
               # print(bool_result)
                line.strip()    
                line =str(line)
                line = line[1:]
                line =line[1:len(line)-1]
                line =  line.replace('\\t',' ')
                line =  line.replace('\\n',' ')
                line = line.split(' ')
                same= False 
                emb1,emb2=[] ,[]
                if len(line) == 4:
                    same =True
                    file_name = line[0]
                    folder_path = data_dir+'\\'+file_name+'\\'
                    files = [f for f in os.listdir( folder_path ) if  isfile(join(folder_path,f)) ]
                    i=1
                    
                    for file in files: 
                       if str(i) == line[1]:
                          
                           path = join( folder_path,file)
                           image = cv2.imread(path)
                           res = core.pipline_image(image,preprocess=False,clean=False,draw=False)
                           if res['result'] :
                               emb1 =res['embeddings']
                           else :
                               emb1= []
                       if str(i) == line[2]:
                           path = join( folder_path,file)
                           image = cv2.imread(path)
                           res = core.pipline_image(image,preprocess=False,clean=False,draw=False)
                           if res['result'] :
                               emb2 =res['embeddings']
                           else :
                               emb2= []
                       i=i+1
                   
                elif len(line) == 5:
                    file_name = line[0]
                    folder_path = data_dir+'\\'+file_name+'\\'
                    files = [f for f in os.listdir( folder_path ) if  isfile(join(folder_path,f)) ]
                    i=1
                    for file in files: 
                       if str(i)== line[1]:
                           path = join( folder_path,file)
                           image = cv2.imread(path)
                           res = core.pipline_image(image,preprocess=False,clean=False,draw=False)
                           if res['result'] :
                               emb1 =res['embeddings']
                           else :
                               emb1= []
                    file_name = line[2]
                    folder_path = data_dir+'\\'+file_name+'\\'
                    files = [f for f in os.listdir( folder_path ) if  isfile(join(folder_path,f)) ]
                    i=1
                    for file in files: 
                       if str(i) == line[3]:
                           path = join( folder_path,file)
                           image = cv2.imread(path)
                           res = core.pipline_image(image,preprocess=False,clean=False,draw=False)
                           if res['result'] :
                               emb2 =res['embeddings']
                           else :
                               emb2= []
                if emb1 ==[] or emb2==[]:
                    ret =False
                else :
                    ret = core.verificate(emb1,emb2,eps)
               # print(ret , '-----' , same)
                if same : 
                    if not ret ==same :
                        FRR=FRR+1
                if not same : 
                    if not ret ==same : 
                        FAR=FAR+1
                if ret==same:
                    #print(True)
                    bool_result.append('1')
                else:
                    #print(False)
                    bool_result.append('0')
            
            bool_result =np.array(bool_result)

            bool_result,count = np.unique(bool_result,return_counts=True)

            print ('FRR', (FRR/j))
            print ('FAR', (FAR/j))
            print('Acc', (count[1])/j)
            print('--------------------')
            return ((count[1])/j),(FRR/j),(FAR/j)
                
#%%

steps = [0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3]

options = {'detect_model' : 'mtcnn', 'align_model':'2d' ,'recognize_model' : 'arcface','track_model' : 'sort','prediction_model':'knn'}
core.set_pipline_option(model_options = options)
base = 0.4
step = 0.1
all_frr=[]
all_far=[]
all_acc=[]
for i in range(1,11):
    eps= step*i +base
    ACC,FRR,FAR = validate_lfw(eps=eps)
    print('start with eps ', eps)
    all_frr.append(FRR)
    all_far.append(FAR)
    all_acc.append(ACC)
cdict = {'FAR': 'red', 'FRR': 'blue', 'ACC': 'green'}
fig, ax = plt.subplots()
fig.suptitle('mtcnn - facenet tests')
plt.xlabel('step')
plt.ylabel('precentage')
ax.scatter(steps, all_frr)
ax.scatter(steps, all_far)
ax.scatter(steps, all_acc)
ax.plot(steps, all_frr, c = cdict['FRR'], label = 'FRR')
ax.plot(steps, all_far, c = cdict['FAR'], label = 'FAR')
ax.plot(steps, all_acc, c = cdict['ACC'], label = 'ACC')
ax.legend()
plt.show()


