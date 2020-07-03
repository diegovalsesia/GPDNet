import argparse
import h5py
import numpy as np
import os
import sys
import re
from Config import Config
from net_dn import Net
from C2C_distance import compute_C2C

 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

parser = argparse.ArgumentParser()

parser.add_argument('--log_dir', default='log', help='Log directory')
parser.add_argument('--save_dir', default='Saved Model', help='Trained model directory')
parser.add_argument('--start_iter', type=int, default=1, help='Start iteration (ex: 10001)')

parser.add_argument('--train_data_file', default='', help='Training data file')
parser.add_argument('--clean_val_data_file', default='', help='Validation data file')
parser.add_argument('--noisy_val_data_file', default='', help='Validation data file')

param = parser.parse_args()

config = Config()

config.log_dir = param.log_dir
config.save_dir = param.save_dir

config.train_data_file = param.train_data_file

# import train data
print('Loading training files')
f = h5py.File(config.train_data_file, 'r')
data = f['data'][:]
data = (data.astype(np.float))
patches_clean = data


# import val data
# print('Loading validation files')
# with h5py.File(config.val_data_file) as dataset_val:
#     val_data = dataset_val['data'][:]
#     val_data= val_data.astype(np.float)
#     clean_val_batch = val_data
#     noisy_val_batch = clean_val_batch +  np.random.normal(0,config.sigma, clean_val_batch.shape)
#     clean_val_batch = clean_val_batch[:config.batch_size_validation,:,:]
#     noisy_val_batch = noisy_val_batch[:config.batch_size_validation,:,:]

print('Loading validation files')
f_noisy_val = h5py.File(param.noisy_val_data_file, 'r')
data = f_noisy_val['data'][:]
noisy_val_batch = data.astype(np.float)
print(noisy_val_batch.shape)
f_noisy_val.close()

f_clean_val = h5py.File(param.clean_val_data_file, 'r')
data = f_clean_val['data'][:]
clean_val_batch = data.astype(np.float)
print(clean_val_batch.shape)
f_clean_val.close()

clean_val_batch = clean_val_batch[:config.batch_size_validation,:,:]
noisy_val_batch = noisy_val_batch[:config.batch_size_validation,:,:]

model = Net(config)
model.do_variables_init()


if param.start_iter == 1:
    start_iter = 0
else:
    start_iter = param.start_iter
    model.restore_model(config.save_dir +'model.ckpt')
    print('Resumed training from iter %d' % start_iter)



# training
for iter_no in range(start_iter, config.N_iter):
      pos = np.random.choice(patches_clean.shape[0], size=config.batch_size)
      clean_batch = patches_clean[pos, :, :]
      noisy_batch = clean_batch + np.random.normal(0, config.sigma, clean_batch.shape)
  
      # train
      model.fit(clean_batch, noisy_batch, iter_no)
      
      # validate
      if iter_no % config.validate_every_iter == 0:
        model.validate(clean_val_batch, noisy_val_batch, iter_no)

        
    	# save model
      if iter_no % config.save_every_iter == 0:
        model.save_model(config.save_dir+'model.ckpt')
        with open(config.log_dir+'start_iter', "w") as text_file:
  			     text_file.write("%d" % iter_no)
                                       
    	# backup model
      if iter_no % 10000 == 0:
    		os.mkdir(config.save_dir+str(iter_no))
    		model.save_model(config.save_dir+str(iter_no)+'/model.ckpt')
    

