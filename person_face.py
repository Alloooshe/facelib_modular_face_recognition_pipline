
class Face(object):
   
     __slots__ = ['num','name','emb', 'bboxes',  'landmarks']
     def __init__(self,num,bboxes,points, emb,name='unknown'):
        self.num = num
        self.name = name
        self.bboxes = bboxes
        self.landmarks = points
        self.emb = emb

