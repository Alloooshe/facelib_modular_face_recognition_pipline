import numpy as np 


class FaceMatch():
    '''
    
    FaceMatch is the class wrapping all face matching process 
    
    Attributes:
        data (list of Face) : comparision data 
        kdtree_model (kdtree model) 
        xgboost_model (xgboost model)
        kmeans_model (kmeans model)
        kmeans_labels (list of string) : the names of clusters
        isloaded_xgboost (boolean) : True if model is loaded 
        isloaded_kdtree (boolean) : True if model is loaded 
        isloaded_kmeans (boolean) : True if model is loaded 
    '''
    def __init__(self):
        self.data = None
        self.kdtree_model=None
        self.isloaded_kdtree=False
        self.kmeans_model =None
        self.isloaded_kmeans = False 
        self.kmeans_labels =None

    def match_face_kdtree(self,sample,  eps=0.3,k=5,r=0.5,threshold= 0.2):
        '''
        match sample with data using kdtree model 
        
        Args: 
            sample (numpy array) : the embeddings of the target 
            eps (float) : used to detect unkown condition if the two closest  prediction are smaller than eps it detects unkwon condition
            k (int) : number of predictions required default is 2 
            r (float) : radius of predictions 
            threshold (float) : threshold to reject prediction 
        Retuns:
            name (string) : identity of target 
            int index of the matched person ion the data set 
            float distance between target and match
        '''
        ind,dist = self.kdtree_model.query_radius(sample, r=r,return_distance =True,sort_results =True)
        ind= ind[0]
        dist = dist[0]
        if len(ind)==0:
            return ["unkown"],-1,-1
        if len(ind) ==1 :
            n1 = self.get_name_from_data(ind[0])
            return [n1],ind[0],dist[0]
        
        n1 = self.get_name_from_data(ind[0])
        n2 = self.get_name_from_data(ind[1])
        n3 = self.get_name_from_data(ind[2]) 
        n4 = self.get_name_from_data(ind[3]) 
        n5 = self.get_name_from_data(ind[4]) 
        if n1 ==n2 : 
            return [n1,n2,n3,n4,n5],ind[0],dist[0]       
        ratio=dist[0]/dist[1]
        if abs(ratio-1) <eps:
            return [n1,n2,n3,n4,n5],ind[0],dist[0]
        else:
            return ["unkown",n1,n2,n3,n4],-1,-1


    
    def match_face_knn(self,sample,eps=0.1):
       '''
         match sample with data using knn method
         
         Args :
             sample (numpy array) : the embeddings of the target 
             eps (float) : the eps to detect unkown matching 
             
         Retuns:
            name (string) : identity of target 
            int index of the matched person ion the data set 
            float distance between target and match
       '''
       dist = self.calculate_dist(sample)
       ind = dist.argsort()
       names = []
       name =self.get_name_from_data(ind[0])
       n1= self.get_name_from_data(ind[1])
       n2 = self.get_name_from_data(ind[2])
       n3 = self.get_name_from_data(ind[3])
       n4 = self.get_name_from_data(ind[4])
       names.append(n1)
       names.append(n2)
       names.append(n3)
       names.append(n4)
       print(name ,'---',n1,'--',n2,'---',n3,'---',n4)
    #   print('dist' ,(dist[ind[1]]-dist[ind[0]]))
       
       for i in range(0,2): 
           if not names[i] ==name: 
               if (dist[ind[i+1]]-dist[ind[0]]) < eps : 
                   return ['unkown',name,n1,n2,n3],-1,-1
               
  
       return [name,n1,n2,n3,n4],ind[0],dist[ind[0]]
           
    def match_face_kmeans(self,sample,eps=0.01):
        '''
         match sample with data using kmeans model 
         
         Args :
             sample (numpy array) : the embeddings of the target 
         Retuns:
            name (string) : identity of target 
            int index of the matched person ion the data set 
            float distance between target and match
        '''
        sample =prepare_sample_kmeans(sample)
        pred = self.kmeans_model.predict(sample)
        dist=calculate_dist(sample,data =self.kmeans_model.cluster_centers_ )
        dist = np.array(dist)
        indx = dist.argsort()[:2]
        print(pred)
        pred = pred[0]

        
        if abs( dist[indx[0]]-dist[indx[1]] )<eps:
           return  ['unkown'],-1,-1
        
        return [self.kmeans_labels[pred]] , pred,-1

    def match_face(self,sample,model_name):
        '''
        wrapper function to perfom matching 
        
        Args : 
            sample (numpy array) : the embeddings of the target 
            model_name (string) : name of the model to use in matching 
        
        Returns : 
            name (string) : identity of target 
            int index of the matched person ion the data set 
            float distance between target and match
            
        '''
        if model_name =="kdtree":
           if not self.isloaded_kdtree: 
               self.kdtree_model=load_kdtree()
               self.isloaded_kdtree =True 
               
           return self.match_face_kdtree(sample)
       
        elif model_name=="knn":
            return self.match_face_knn(sample)
        elif model_name =="kmeans":
            if not self.isloaded_kmeans:
                self.isloaded_kmeans =True
                self.kmeans_model,self.kmeans_labels = load_kmeans()
            return self.match_face_kmeans(sample)
            
           
    def match_image(self,emb,model_name) : 
        sample = self.prepare_emb_sample(emb)
        name,ind,dist = self.match_face(sample,model_name)
        return name,5
        
    def voting_recognition(self,emb,model_name, prev=[]):
        '''
        uses voting method to perfome matching  on image 
        
        Args : 
            emb (numpy array) : sample embeddings
            model_name (string) : name of the mathcing model to use 
            prev(list of string) : names of prevoius matchings 
        Returns : 
            string name of the person 
            int number of votes for this name
            
        '''
        sample = self.prepare_emb_sample(emb)
        name,ind,dist = self.match_face(sample,model_name)
        name=name[0]
        x=prev
        x.append(name)
        unique, counts = np.unique(x, return_counts=True)
        indx = np.argmax(counts)
        return unique[indx],counts[indx]
    
    def featurefusion_recognition(self,emb,model_name,prev=[]):
        '''
        uses feature fusion  method to perfome matching  on image 
        
        Args : 
            emb (numpy array) : sample embeddings
            model_name (string) : name of the mathcing model to use 
            prev(list of string) : names of prevoius matchings 
        Returns : 
            string name of the person 
            int number of votes for this name
            
        '''
        prev.append(emb)
        emb = np.mean(prev,axis=1)
        name,ind,dist = self.match_face(emb,model_name)
        return name[0],dist
    
    def prepare_emb_sample(self,sample):
        '''
        helper funciton to prepare sample for matching chaing its shape 
        
        Args: 
           sample (numpy array) : the embeddings of the target 
           
        Returns :
            q (numpy array)  : processed sample 
        
        '''
        q=np.reshape( sample,(1,-1) )
        return q
   
    def get_name_from_data(self,ind=0):
        '''
        helper funciton to get the name of sample in data using its index 
        
        Args:
            ind (int) : the index of the sample
        
        Returns:
            string name of the sample in the index
        
        '''
        return self.data[ind].name
   
    
    def calculate_dist(self, x, data=False,metric = "ecl"):
      """
      caluclate distance between two embeddings.
      
      Args:
          x (numpy array): the fist embedding to considre
          data (numpy array) : array of embeddings to which we should compare.
     
      Return:
          dist (nunpy array) : distances from each data point to tthe target.
      """
      if not data :
          data=self.data
          data = np.asarray([ x.emb[0] for x in data])
      dist = []
      for d in data :
            if metric =="ecl":
               tmp =  np.sqrt(np.sum(np.square(np.subtract(  x, d ) )))
           # elif metric =="acos" : 
                
            dist.append(tmp)
      dist = np.squeeze(dist)
      return dist
