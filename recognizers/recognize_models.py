'''

this file containes helper functions to train,save and load prediction models used in matching 

'''

from sklearn.neighbors import KDTree
import pickle
import os
import numpy as np 
from sklearn.cluster import KMeans

def train_kdtree(data,leaf_size=2,save_tree=True, save_dir=os.path.dirname(os.path.realpath(__file__)) +'\\models\\match\\'):
    '''
    train kdtree model and save it to the disk as .pkl file 
    
    Args:
        data (list of Face) : list of face instance containes informations about the data you want to use to train kdtree model
        leaf_size (int) : the min leaf size of kdtree model 
        save_tree (boolean) : if True save the trained model to the disk 
        save_dir (string) : the saving diracoty 
    
    Returns:
        tree (kdtree model) : the trained model
     
    '''
    tree = KDTree(data, leaf_size=leaf_size)
    if save_tree:
        save_dir = os.path.join(save_dir, os.path.basename(save_dir +'kdtree.pkl') ) 
       
        with open(save_dir, 'wb') as outfile:
            pickle.dump(tree, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    return tree

def prepare_face_data(data):
    '''
    helper function to prepare data . it extracts embeddings from list of Face
    
    Args:
        data (list of Face)
    
    Returns:
        emb (numpy array) : the embeddings extracted from input
        
    '''
    
    emb =np.asarray([ x.emb[0] for x in data])
    emb=np.squeeze(emb)
    return emb 
 
def load_kdtree(model_dir= os.path.dirname(os.path.realpath(__file__)) +'\\models\\match\\kdtree.pkl'):
    '''
    loads kdtree model 
    
    Args : 
        model_dir (string) :  the saved model diracoty
    
    Returns:
        tree (kdtree model) : the saved model 
    
    '''
    print('loading kdtree model ...')
    with open (model_dir,'rb') as outfile:
            tree =  pickle.load(outfile)
    return tree


def train_kmeans(data,n_clusters , names,save_dir=os.path.dirname(os.path.realpath(__file__)) +'\\models\\match\\' ):
    '''
    train and save kmenas culstering model 
    
    Args: 
        data (list of Face) : input data  
        n_clusters (int) : number of required cluster should be equal or greater than  number of people in data set
        names (list of string) : names of people in data set 
        save_dir (string) : the saving diracoty 
        
    Returns : 
        True if training was successfull
    '''
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    model_dir = os.path.join(save_dir, os.path.basename(save_dir +'kmeans.pkl') ) 
    with open(model_dir, 'wb') as outfile:
        pickle.dump(kmeans, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    labels =[]
    for x in kmeans.cluster_centers_:
        dist = calculate_dist(x,data)
        indx = dist.argsort()
        tmp = [names[indx[0]],names[indx[1]],names[indx[2]],names[indx[3]],names[indx[4]]]
        print(tmp)
        labels.append(names[indx[0]])
    save_dir = os.path.join(save_dir, os.path.basename(save_dir +'kmeans_labels.pkl') ) 
    with open(save_dir, 'wb') as outfile:
        pickle.dump(labels, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    return True 

def prepare_kmeans(data):
    '''
     prepare data for kmeans culstering 
     
     Args: 
         data (list of Face) : the data to use 
    
    Returns : 
        emb (numpy array) : data embeddings
        names (list of string) : the names of people in the data set 
        int (int) : the count of different people in the dataset  
    '''
    emb =[x.emb for x in data]
    names =[x.name for x in data ]
    count = np.unique(names, return_counts=False)
    emb =np.squeeze( np.array(emb) )
    return emb,names,len(count)

def prepare_sample_kmeans(sample):
    '''
    prepare sample for kmeans prediction basicly chaning its shape 
    
    
    Args : 
        sample (numpy array) : the input sample 
    
    Returns :
        sample (numpy array) : the processed sample 
        
    '''
    sample =np.reshape(sample,(sample.shape[1],1))
    sample = np.transpose(sample)
    return sample


def load_kmeans(model_dir= os.path.dirname(os.path.realpath(__file__)) +'\\models\\match\\kmeans.pkl',names_dir = os.path.dirname(os.path.realpath(__file__)) +'\\models\\match\\kmeans_labels.pkl' ):
    '''
    load kmeans model and culsters labels 
    
    Args: 
       model_dir (string) : saved model diractory 
       names_dir (string) : saved labels diractory 
    
    Returns: 
        model (kmeans model) : the saved model 
        lables (list of string) : clusters labels 
        
    '''
    print('loading kmeans model ...')
    with open (model_dir,'rb') as outfile:
            model =  pickle.load(outfile)
    with open (names_dir,'rb') as outfile:
            labels =  pickle.load(outfile)
    return model,labels


def calculate_dist( x, data=False,metric = "ecl"):
      """
      caluclate distance between two embeddings.
      
      Args:
          x (numpy array): the fist embedding to considre
          data (numpy array) : array of embeddings to which we should compare.
     
      Return:
          dist (nunpy array) : distances from each data point to tthe target.
      """
      dist = []
      for d in data :
            if metric =="ecl":
               tmp =  np.sqrt(np.sum(np.square(np.subtract(  x, d ) )))
           # elif metric =="acos" : 
                
            dist.append(tmp)
      dist = np.squeeze(dist)
      return dist
    