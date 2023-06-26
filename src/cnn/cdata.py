import os
import re
import json

default_re = re.compile(r'.(jpg|png)$')

class CData:
    def __init__(self):
        return
    
    def __scan(self, path, filter=default_re):
        files = os.listdir(path)
        ret = []
        for x in files:
            _path = os.path.join(path, x)
            if os.path.isdir(_path):
                ret += self.__scan(path)
            elif filter==None or re.search(filter, x)!=None:
                ret.append(_path)
        return ret
    
    def read_folder(self, path):
        folders = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
        return {x: self.__scan(os.path.join(path, x)) for x in folders}
    
    def from_dict(self, my_dict):
        features = []
        labels = []
        for key, value in my_dict.items():
            for _path in value:
                features.append(_path)
                labels.append(int(key))
        return (features, labels)

    def read_config(self, path):
        with open(path, 'r') as f:
            json_data = f.read()
        return json.loads(json_data)
    