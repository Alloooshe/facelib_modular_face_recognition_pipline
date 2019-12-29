import abc
class FaceDetector(object):
    '''
    FaceDetector is an abstract calss for face detectors to implement. it gurantees a degree of consistency when using different models in the detection process.
    
    Functions : 
        name 
        detect
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
        raise NotImplementedError('fucntion \"name\" was not implemented in ' + self.NAME)

    @abc.abstractmethod
    def detect(self, img):
         raise NotImplementedError('fucntion \"detect\" was not implemented in ' + self.NAME)
    @abc.abstractmethod
    def load(self):
        raise NotImplementedError('fucntion \"load\" was not implemented in ' + self.NAME)
        
    @abc.abstractmethod
    def configure(self):
        pass
    
    @abc.abstractmethod
    def parse_output(self):
        pass
    
    @abc.abstractmethod
    def clean(self):
        raise NotImplementedError('fucntion \"clean\" was not implemented in ' + self.NAME)