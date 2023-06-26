from my_data_pair import MyData
import tensorflow as tf
import os
from tensorflow.image import ssim
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from shape_loss import shape_loss
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

class NumpyDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super(NumpyDecoder, self).__init__(object_hook=self.dict_to_object, *args, **kwargs)

    def dict_to_object(self, d):
        if '__ndarray__' in d:
            dtype = np.dtype(d['__ndarray__']['dtype'])
            data = np.array(d['__ndarray__']['data'], dtype=dtype)
            return data
        return d

class __IconGroup:
    def __init__(self):
        return
    
    def conf(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.save_dir_matrix = os.path.join(self.save_dir, 'matrix.txt')
        self.save_dir_labels = os.path.join(self.save_dir, 'labels.txt')
    
    def prepare(self, dataset_train, total_num, isload):
        if(isload and os.path.exists(self.save_dir_matrix) and os.path.exists(self.save_dir_labels)):
            print('[STEP]...load matrix')
            return self.load()
        else:
            matrix, labels = self.distance_matrix(dataset_train,total_num)
            print('[STEP]...save matrix')
            self.save(matrix, labels)
            return matrix, labels
    
    def hierarchical_clustering(self, source_path, source_config_path, isload=True, num_clusters = 100):
        print('[STEP]...start build data data groups: ', num_clusters)
        my_data = MyData(1)
        my_data.read(source_path, source_config_path)
        dataset_train = my_data.create_dataset()
        # 计算距离矩阵'
        total_num = my_data.num_label
        print('[STEP]...calculate distance matrix')
        matrix, labels = self.prepare(dataset_train,total_num, isload)
            
        # 创建AgglomerativeClustering对象，并进行聚类
        print('[STEP]...clustering')
        clustering = AgglomerativeClustering(n_clusters=num_clusters)
        cluster_labels = clustering.fit_predict(matrix)
        print(cluster_labels)
        cluster_dict = {}
        groups = {}
        for index, value in enumerate(cluster_labels):
            if not value in cluster_dict:
                cluster_dict[value]=[]
                groups[value]=[]
            cluster_dict[value].append(index) # index = the index of tensor dataset | label[index] = real index
            # groups[value].append(labels[index])
        presentors = {}
        for key, group in cluster_dict.items(): #[1,3,4]
            cluster_dict[key] = self.sort_representor_min(group, matrix) #sort
            #print(cluster_dict[key])
            groups[key] = [labels[x] for x in cluster_dict[key]]
            presentors[key] = groups[key][0]
            # presentors[key] = labels[self.choose_representor_min(group, matrix)]
        print('[STEP]...finsihed!')
        return presentors, groups
    
    def save_list_to_file(self, data, file_path):
        with open(file_path, 'w') as file:
            json.dump(data, file, cls=NumpyEncoder)

    def load_list_from_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file, cls=NumpyDecoder)
            return data
        
    def save(self, matrix, labels):
        self.save_list_to_file(matrix, self.save_dir_matrix)
        self.save_list_to_file(labels, self.save_dir_labels)
        
    def load(self):
        matrix = self.load_list_from_file(self.save_dir_matrix)
        labels = self.load_list_from_file(self.save_dir_labels)
        return matrix,labels

    def sort_representor_min(self, group, matrix):
        dic = []
        for v1 in group:
            sum = 0 
            for v2 in group:
                sum += matrix[v1][v2]
            dic.append(sum)
        #print(group, dic)
        return [x for _, x in sorted(zip(dic, group))]
    
    def choose_representor_min(self, group, matrix):
        min_value = float('inf')
        min_index = 0
        for v1 in group:
            sum = 0 
            for v2 in group:
                sum += matrix[v1][v2]
            if(sum < min_value):
                min_value = sum
                min_index = v1
        return min_index
        
    def distance_matrix(self, dataset_train, total):
        index_labels =  [0 for _ in range(total)]
        matrix = [[0 for _ in range(total)] for _ in range(total)]
        for index1, (image1, label1) in dataset_train.take(total).enumerate():
            _index1 = index1.numpy()
            index_labels[_index1] = label1.numpy()[0]
            for _index2, (image2, label2) in dataset_train.take(total).enumerate():
                _index2 = _index2.numpy()
                if _index1 != _index2:
                    matrix[_index1][_index2] = matrix[_index2][_index1] = self.distance(image1, image2)
                print(f'distance {_index1}/{total} - {_index2}/{total}  finished!', end="\r")
        return matrix, index_labels
    
    def distance(self, image1, image2):
        return (shape_loss.range_0_1(shape_loss.ssim(image1, image2)) + shape_loss.range_0_1(shape_loss.chamfer_loss(image1, image2)) + shape_loss.range_0_1(shape_loss.iou_loss(image1, image2))).numpy()

icon_group = __IconGroup()