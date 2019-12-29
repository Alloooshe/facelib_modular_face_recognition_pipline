import abc


class FaceAligner(object):
    '''
    FaceAligner is an abstract class for aligners to inherite 
    
    Functions: 
        name 
        configure
        align
        parse_ouput
    
    '''
    def __init__(self):
        pass

    def __str__(self):
        return self.name()

    @abc.abstractmethod
    def name(self):
        return 'embedder'



    @abc.abstractmethod
    def configure(self):
        pass

    @abc.abstractmethod
    def align(self, img):
        pass
    
    @abc.abstractmethod
    def parse_output(self):
        pass
        


