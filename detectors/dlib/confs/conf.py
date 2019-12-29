import os
import yaml


class DeepFaceConfs:
    __instance = None

    @staticmethod
    def get():
        """ Static access method. """
        if DeepFaceConfs.__instance is None:
            DeepFaceConfs()
        return DeepFaceConfs.__instance

    def __init__(self):
        if DeepFaceConfs.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            DeepFaceConfs.__instance = self

        dir_path = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(dir_path, 'basic.yaml'), 'r') as f:
            self.conf = yaml.load(f)

    def __getitem__(self, key):
        return self.conf[key]
