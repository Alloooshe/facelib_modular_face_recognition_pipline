import abc


class FaceRecognizer(object):
    '''
    FaceRecognizer is an abstract calss for face recognizers(embedders) to implement
     
    Functions : 
        name
        extract_features
        load
        configure
        parse_ouput
        clean
    '''
    def __init__(self):
        pass

    def __str__(self):
        return self.name()

    @abc.abstractmethod
    def name(self):
        return 'embedder'



    @abc.abstractmethod
    def load(self):
        raise NotImplementedError('users load fucntion to use this base class')

    @abc.abstractmethod
    def configure(self):
        pass

    @abc.abstractmethod
    def extract_features(self, img):
        pass
    
    @abc.abstractmethod
    def parse_output(self):
        pass
        
    @abc.abstractmethod
    def clean(self):
        pass

