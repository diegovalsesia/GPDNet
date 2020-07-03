from scipy import spatial
import h5py
import os
import numpy as np


#namefile is an h5 file

def knn_matrix_from_file(path, namefile, k):

    with h5py.File(path+ namefile) as dataset_val:
        data = dataset_val['data'][:]
    data=np.asarray(data)
    data = np.squeeze(data)
    point_count=data.shape[0]
    print("Numero di punti: ", point_count)
    
    
    tree=spatial.KDTree(data, leafsize=1024)
    
    [_,i]=tree.query(data, k)
    
    
    return i

def knn_matrix_from_data2(data, k, name):
    # data do not has to be divided in patches
    point_count=data.shape[0]
    print("Numero di punti: ", point_count)
    tree=spatial.KDTree(data, leafsize=1024)
    
    [_,i]=tree.query(data, k)

    with h5py.File('./knn_' + name + '.h5', "w") as ff:
        ff.create_dataset(name='data', data=i, maxshape=(point_count, k), chunks=True)
    
    return i



def knn_matrix_from_data(data, k):
    # data do not has to be divided in patches
    point_count=data.shape[0]
    print("Numero di punti: ", point_count)
    tree=spatial.cKDTree(data, leafsize=30)
    
    [_,i]=tree.query(data, k)
    
    return i




