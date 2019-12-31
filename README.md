# facelib_modular_face_recognition_pipline
simple and modular framework to perform all stages of face recognition pipeline (detection, alignment, embedding, matching) with the possibility to add and use different models and techniques for each stage.

## content
* what is facelib
* architecture
* video recognition
* usage
* acknowledgment



![friends]( https://github.com/Alloooshe/facelib_modular_face_recognition_pipline/blob/master/images/out.jpg)

## what is facelib ?
the project solves the problem of face recognition by solving each of the following problems individually: face detection, face alignment, face embedding and face matching or recognition, the code reflects the academic partition so that each step can be carried out independently with whatever framework or programming language the user prefers and then it will be integrated in the pipeline. the project also works with video or live stream and implement object tracking to get better results and to reduce the computation cost when dealing with high frame rate.
the project aims to find a software solution that allows developers to quickly customize a face recognition pipeline depending on their needs, also the project help researchers evaluate their models developed for a specific task (such as face embedding) with different detection and alignment methods. there are many other useful use cases for the project. 
the code is organized and documented in a good way so that developers can read and make adjustments relatively easy, the docs is built using sphinx so the user can search for any needed documentation. 
## architecture
![facelib architecture]( https://github.com/Alloooshe/facelib_modular_face_recognition_pipline/blob/master/images/architecture.PNG)

The code architecture consists of independent units that perform independent tasks, the main block is the class FaceCore which handles the pipeline operation but can be used for single tasks such as detection or embedding. 
Each class has the option to “clean” the model it leaded, the FaceCore is built to keep the models in memory in order to avoid loading overhead. This boost the performance of facelib in applications. In any time, you can use the clean option to free memory up. 


## video recognition
The FaceCore class can perform video face recognition and tracking using the function “process stream” (which can handle both live stream and video) , this includes ignoring small faces that appears in a video and once a face is bigger than a threshold it then performs recognition pipeline on that face for N (a variable that you can control) frames and it accumulate the decision made about the identity of the face and makes a final decision, once a final decision is made no more recognition is performed and the function continue to only track the face.  
the decision accumulation can be made in two ways (found to have similar performance) which are voting and feature fusion, in voting each of the first N frames has one vote on the final decision while in feature fusion we take the mean of the N embedding vectors extracted in the N frames for a certain face and then make a final matching using the new embedding vector.
This is important to reduce the computation cost and to make better face recognition. 


 ## usage
coming soon

## demo
[face recognition video]( https://www.youtube.com/watch?v=kSNk_1QLzbQ)

The faceapp.ui contains a  UI generated using PyQt designer (it can be comipled to different platforms) the faceapp.py the python compiled version. You can run the faceapp.py and try image, video and webcam face recognition, you can also add faces to your database. 
## acknowledgment
1. [MTCNN](https://github.com/ipazc/mtcnn)
2. [facenet](https://github.com/davidsandberg/facenet)
3. [SORT](https://github.com/abewley/sort)
4. [arcface](https://github.com/deepinsight/insightface)

