import argparse
import h5py
import numpy as np
import os
import sys
import re
from Config import Config
from net_test_conv import Net
from C2C_distance import compute_C2C
from knn_matrix import knn_matrix_from_data
import scipy.io as sio


def testing(mydata_test, mynoisy_test_patches, matrix, name,number_model,  sigma, folder):
    bad_data = mynoisy_test_patches
    myx_hat = np.zeros_like(mynoisy_test_patches)
    n_points=bad_data.shape[0]*bad_data.shape[1]
    myx_hat = model.denoise(bad_data, matrix, n_points)

    myx_hat= np.reshape(myx_hat, [-1,3])
    point_count = myx_hat.shape[0]            
    with open(folder+ '/'+name+'_'+str(number_model)+".xyz", 'w') as xyzfile:
      for index in range(0, int(point_count)):
          xyzfile.write(' '.join(map(str, myx_hat[index])))  
          xyzfile.write('\n')

    C2C= compute_C2C(mydata_test, myx_hat, './'+folder+'/', save_var=False)
    C2C=C2C*1e+06
    with open(folder+'/c2c_' + name+"_mse+c2c_"+sigma+".txt", 'a') as textfile:
       textfile.write(str(C2C))
       textfile.write('\n')
    print 'C2C: %.5f' % (C2C)
    
    return myx_hat
 

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='DenoisePointNet.py', help='Model name: net')
parser.add_argument('--save_dir', default='Saved Model', help='Trained model directory')
parser.add_argument('--denoised_dir', default='', help='Testing results data directory')
parser.add_argument('--gt_dir', default='./Dataset/Test_Shapenet_h5/gt/', help='Testing gt data directory')
parser.add_argument('--noisy_dir', default='./Dataset/Test_Shapenet_h5/noisy/', help='Testing noisy data directory')



param = parser.parse_args()

config = Config()
config.save_dir = param.save_dir
folder = param.denoised_dir


print('Define Model')
model = Net(config)
model.do_variables_init()

model.restore_model(config.save_dir + 'model.ckpt')

# import TEST dataset
print('Loading test file')
cat = ['airplane', 'bench', 'car', 'chair', 'lamp', 'pillow', 'rifle', 'sofa', 'speaker', 'table']


for i in cat:
    print(i)
    # Groud-true 
    with h5py.File(param.gt_dir+i+'.h5') as dataset_val:
        all_clean_test = dataset_val['data'][:]
        all_clean_test= all_clean_test.astype(np.float)
    print(all_clean_test.shape)
    all_clean_test=np.asarray(all_clean_test)

    #Noisy
    with h5py.File(param.noisy_dir+i+'_'+str(config.sigma)+'.h5') as dataset_val:
        all_noisy_test = dataset_val['data'][:]
        all_noisy_test= all_noisy_test.astype(np.float)
    print(all_noisy_test.shape)
    all_noisy_test=np.asarray(all_noisy_test)



    for j in range (all_noisy_test.shape[0]):
        clean_test = all_clean_test[j, : , :]
        noisy_test = all_noisy_test[j, :, :]
        print(clean_test.shape)
        clean_test= clean_test.astype(np.float32)
        noisy_test= noisy_test.astype(np.float32)

        print("Computing knn_matrix")
        test_nn_matrix= knn_matrix_from_data(noisy_test, config.knn)

        print("Computing c2c distance")
        C2C_noisy = compute_C2C(clean_test, noisy_test)
        C2C_noisy = C2C_noisy*1e+06
        print 'MSE Noisy Shapenet %s : %.5f' % (i, C2C_noisy)

        if j == 0:
            with open(folder+'/c2c_'+i+'_noisy_'+str(config.sigma)+'.txt', 'w') as textfile:
                textfile.write(str(C2C_noisy))
                textfile.write('\n') 
        else:
            with open(folder+'/c2c_'+i+'_noisy_'+str(config.sigma)+'.txt', 'a') as textfile:
                textfile.write(str(C2C_noisy))
                textfile.write('\n') 


        # make the test point cloud in patches
        noisy_test_patches = np.reshape(noisy_test, [-1, 1024, 3])
        noisy_test_patches= noisy_test_patches.astype(np.float32)
        noisy_test= noisy_test.astype(np.float32)

        dn = testing(clean_test,noisy_test_patches, test_nn_matrix,i,j, str(config.sigma), folder)



