import tensorflow as tf
import numpy as np
from C2C_distance import compute_C2C
from knn_matrix import knn_matrix_from_data
import scipy.io as sio

class Net:

    def __init__(self, config):
        self.config = config
        self.N = config.N

        ######### not running out gpu sources ##########
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)

        ############### define variables ###############
        self.W = {}
        self.b = {}
        self.scale = {}
        self.beta = {}
        self.pop_mean = {}
        self.pop_var = {}
        self.alpha = {}
        
        self.dn_vars = []
         
        name_block = "pre"
        for i in range (config.pre_n_layers):
              self.W[name_block+"_"+str(i)] = tf.get_variable(name_block+"_"+str(i), [1, config.pre_Nfeat[i], config.pre_Nfeat[i+1]], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
              self.dn_vars = self.dn_vars + [self.W[name_block+"_"+str(i)]]
              self.create_bn_variables_dn(name_block + str(i), config.pre_Nfeat[i+1])
        # residual
        name_block = "residual"
        for i in range(config.n_block):
            for j in range(config.conv_n_layers):
                self.create_gconv_variables_dn(name_block + str(i), j, config.Nfeat, config.prox_fnet_Nfeat, config.Nfeat,config.rank_theta, config.stride, config.stride)
                self.create_bn_variables_dn(name_block + str(i) + "_" + str(j), config.Nfeat)
                # weights for the 1-D convolution self loop
                self.W[name_block + "_sl_" + str(i) + "_" + str(j)] = tf.get_variable(name_block + "_sl_" + str(i)+ "_" + str(j), [1, config.Nfeat, config.Nfeat], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
                self.dn_vars = self.dn_vars + [self.W[name_block + "_sl_" + str(i)+ "_" + str(j)]]           
        #last
        name_block = "last"
        self.create_gconv_variables_dn(name_block, 0, config.Nfeat, config.prox_fnet_Nfeat, config.input_ch,config.rank_theta, config.stride, config.input_ch)
        # weights for the 1-D convolution self loop
        self.W[name_block + "_sl_0"] = tf.get_variable(name_block + "_sl_0",[1, config.Nfeat, 1], dtype=tf.float32,initializer=tf.glorot_normal_initializer())
        self.dn_vars = self.dn_vars + [self.W[name_block + "_sl_0"]]

        ##placeholders
        self.x_clean = tf.placeholder("float", [None, config.patch_size[0], config.patch_size[1]],name="clean_image")
        self.x_noisy = tf.placeholder("float", [None, config.patch_size[0], config.patch_size[1]],name="noisy_image")
        self.is_training = tf.placeholder(tf.bool, (), name="is_training")
        self.is_validation=tf.placeholder(tf.bool, (), name="is_validation")
        self.knn_matrix = tf.placeholder(tf.int32, [None,config.knn], name="knn_matrixx")
        self.number_points=tf.placeholder(tf.int32, (), name="number_points")

        self.__make_compute_graph()


    def create_bn_variables_dn(self, name, Nfeat):

        self.scale['bn_scale_' + name] = tf.get_variable('bn_scale_' + name, [Nfeat], initializer=tf.ones_initializer())
        self.beta['bn_beta_' + name] = tf.get_variable('bn_beta_' + name, [Nfeat],initializer=tf.constant_initializer(0.0))                                               
        self.pop_mean['bn_pop_mean_' + name] = tf.get_variable('bn_pop_mean_' + name, [Nfeat],initializer=tf.constant_initializer(0.0),trainable=False)                                            
        self.pop_var['bn_pop_var_' + name] = tf.get_variable('bn_pop_var_' + name, [Nfeat],initializer=tf.ones_initializer(), trainable=False)                                         
        self.dn_vars = self.dn_vars + [self.scale['bn_scale_' + name], self.beta['bn_beta_' + name]]
        

    def create_gconv_variables_dn(self, name_block, i, in_feat, fnet_feat, out_feat, rank_theta, stride_th1, stride_th2):

        name = name_block + "_nl_" + str(i) + "_flayer0"
        self.W[name] = tf.get_variable(name, [in_feat, fnet_feat], dtype=tf.float32,initializer=tf.glorot_normal_initializer())
        self.b[name] = tf.get_variable("b_" + name, [1, fnet_feat], dtype=tf.float32,initializer=tf.zeros_initializer())
        self.dn_vars = self.dn_vars + [self.W[name], self.b[name]]
        name = name_block + "_nl_" + str(i) + "_flayer1"
        self.W[name + "_th1"] = tf.get_variable(name + "_th1", [fnet_feat, stride_th1 * rank_theta], dtype=tf.float32,initializer=tf.random_normal_initializer(0, 1.0 / (np.sqrt(fnet_feat + 0.0) * np.sqrt(in_feat + 0.0))))
        self.b[name + "_th1"] = tf.get_variable(name + "_b_th1", [1, rank_theta, in_feat], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.W[name + "_th2"] = tf.get_variable(name + "_th2", [fnet_feat, stride_th2 * rank_theta], dtype=tf.float32,initializer=tf.random_normal_initializer(0, 1.0 / (np.sqrt(fnet_feat + 0.0) * np.sqrt(in_feat + 0.0))))
        self.b[name + "_th2"] = tf.get_variable(name + "_b_th2", [1, rank_theta, out_feat], dtype=tf.float32,initializer=tf.zeros_initializer())
        self.W[name + "_thl"] = tf.get_variable(name + "_thl", [fnet_feat, rank_theta], dtype=tf.float32,initializer=tf.random_normal_initializer(0, 1.0 / np.sqrt(rank_theta + 0.0)))
        self.b[name + "_thl"] = tf.get_variable(name + "_b_thl", [1, rank_theta], dtype=tf.float32,initializer=tf.zeros_initializer())
        self.dn_vars = self.dn_vars + [self.W[name + "_th1"], self.b[name + "_th1"], self.W[name + "_th2"],self.b[name + "_th2"], self.W[name + "_thl"], self.b[name + "_thl"]]

        name = name_block + "_" + str(i)
        self.b[name] = tf.get_variable(name, [1, out_feat], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.dn_vars = self.dn_vars + [self.b[name]]
     

    def compute_graph(self, h):
        h = tf.cast(h, tf.float64)
        sq_norms = tf.reduce_sum(h * h, 2)  # (B,N)
        D = tf.abs(tf.expand_dims(sq_norms, 2) + tf.expand_dims(sq_norms, 1) - 2 * tf.matmul(h, h,transpose_b=True))  # (B, N, N)
        D = tf.cast(D, tf.float32)
        h = tf.cast(h, tf.float32)

        return D

    def squared_euclidean_distance(self,point_cloud1, point_cloud2):

        point_cloud1 = tf.cast(point_cloud1, tf.float64)
        point_cloud2 = tf.cast(point_cloud2, tf.float64)
        sq_norms1 = tf.reduce_sum(point_cloud1 * point_cloud1, 2)  # (B,N)
        sq_norms2 = tf.reduce_sum(point_cloud2 * point_cloud2, 2)  # (B,N)
        adjacency_matrix = tf.abs(tf.expand_dims(sq_norms1, 2) + tf.expand_dims(sq_norms2, 1) - 2 * tf.matmul(point_cloud1, point_cloud2,transpose_b=True))  # (B, N, N)
        adjacency_matrix = adjacency_matrix /3
        adjacency_matrix = tf.cast(adjacency_matrix, tf.float32)
        
        return adjacency_matrix


    def myroll(self, h, shift=0, axis=2):

        h_len = h.get_shape()[2]
        return tf.concat([h[:, :, h_len - shift:], h[:, :, :h_len - shift]], axis=2)
        
        
    def dense3(self, h, name):
        return tf.tensordot(h, self.W[name], axes=1) + self.b[name]

   

    def gconv(self, h, name, in_feat, out_feat, stride_th1, stride_th2, compute_graph=True, return_graph=False, D=[]):
    
        if compute_graph:
            D = self.compute_graph(h)
        
        _, top_idx = tf.nn.top_k(-D, self.config.min_nn + 1)  # (B, N, d+1)
        top_idx2 = tf.reshape(tf.tile(tf.expand_dims(top_idx[:, :, 0], 2), [1, 1, self.config.min_nn]),[-1, self.N * (self.config.min_nn)])  # (B, N*d)
        top_idx = tf.reshape(top_idx[:, :, 1:], [-1, self.N * (self.config.min_nn)])  # (B, N*d)
        # K=8*1024, numero di vicini per ogni punto della patch
        x_tilde1 = tf.batch_gather(h, top_idx)  # (B, K, dlm1)
        x_tilde2 = tf.batch_gather(h, top_idx2)  # (B, K, dlm1)
        labels = x_tilde1 - x_tilde2  # (B, K, dlm1)
        tmp=labels
        x_tilde1 = tf.reshape(x_tilde1, [-1, int(in_feat)])  # (B*K, dlm1)
        labels = tf.reshape(labels, [-1, int(in_feat)])  # (B*K, dlm1)
        d_labels = tf.reshape(tf.reduce_sum(labels * labels, 1), [-1, self.config.min_nn])  # (B*N, d)

        name_flayer = name + "_flayer0"
        labels = tf.nn.leaky_relu(tf.matmul(labels, self.W[name_flayer]) + self.b[name_flayer])  # (B*K, F)
        name_flayer = name + "_flayer1"
        labels_exp = tf.expand_dims(labels, 1)  # (B*K, 1, F)
        labels1 = labels_exp + 0.0
        for ss in range(1, int(in_feat / stride_th1)):
            labels1 = tf.concat([labels1, self.myroll(labels_exp, shift=(ss + 1) * stride_th1, axis=2)],axis=1)  # (B*K, dlm1/stride, dlm1)
        labels2 = labels_exp + 0.0
        for ss in range(1, int(out_feat / stride_th2)):
            labels2 = tf.concat([labels2, self.myroll(labels_exp, shift=(ss + 1) * stride_th2, axis=2)],axis=1)  # (B*K, dl/stride, dlm1)
        
        theta1 = tf.matmul(tf.reshape(labels1, [-1, int(in_feat)]),self.W[name_flayer + "_th1"])  # (B*K*dlm1/stride, R*stride)
        theta1 = tf.reshape(theta1, [-1, self.config.rank_theta, int(in_feat)]) + self.b[name_flayer + "_th1"]
        theta2 = tf.matmul(tf.reshape(labels2, [-1, int(in_feat)]),self.W[name_flayer + "_th2"])  # (B*K*dl/stride, R*stride)
        theta2 = tf.reshape(theta2, [-1, self.config.rank_theta, int(out_feat)]) + self.b[name_flayer + "_th2"]
        thetal = tf.expand_dims(tf.matmul(labels, self.W[name_flayer + "_thl"]) + self.b[name_flayer + "_thl"],2)  # (B*K, R, 1)

        x = tf.matmul(theta1, tf.expand_dims(x_tilde1, 2))  # (B*K, R, 1)
        x = tf.multiply(x, thetal)  # (B*K, R, 1)
        x = tf.matmul(theta2, x, transpose_a=True)[:, :, 0]  # (B*K, dl)

        x = tf.reshape(x, [-1, self.config.min_nn, int(out_feat)])  # (N, d, dl)
        x = tf.multiply(x, tf.expand_dims(tf.exp(-tf.div(d_labels, 10)), 2))  # (N, d, dl)
        x = tf.reduce_sum(x, 1)  # (N, dl) *era reduce_mean
        x = tf.reshape(x, [-1, self.N, int(out_feat)])  # (B, N, dl)

        if return_graph:
            return x, D
        else:
            return x


    def gconv_conv(self, points, name, in_feat, out_feat, stride_th1, stride_th2, compute_graph=True, return_graph=False, D=[]):
        #knn_matrix is a matrix of indices
        # points -> [#tot points of pc, neighbors , feature] = [55k, 1024, 66]
        # N=#tot points of pc
        # k = config.min_nn
        #knn_matrix= knn_matrix_from_data(h, self.config.knn)

        #points = tf.gather(h, knn_matrix)

        if compute_graph:
          D = tf.map_fn(lambda feat: self.gconv_conv_inner2(feat), points, parallel_iterations=16, swap_memory=False) # (N,1,knn)
          D = tf.reshape(D,[-1, self.config.knn]) # (N,knn)


        _, top_idx = tf.nn.top_k(-D, self.config.min_nn + 1)  # (N, min_nn+1)
        tmp = top_idx
        #top_idx2 = tf.reshape(tf.tile(tf.expand_dims(top_idx[:, 0], 1), [ 1, self.config.min_nn]),[-1, self.N * (self.config.min_nn)])  # (B, N*d)
        top_idx2 = tf.tile(tf.expand_dims(top_idx[:, 0], 1), [ 1, self.config.min_nn])
        top_idx = top_idx[:, 1:]
        #top_idx = tf.reshape(top_idx[:, 1:], [-1, self.N * (self.config.min_nn)])  # (B, N*d)
        x_tilde1 = tf.batch_gather(points, top_idx)  # (B, K, dlm1) #points
        x_tilde2 = tf.batch_gather(points, top_idx2)  # (B, K, dlm1) #points
        labels = x_tilde1 - x_tilde2  # (B, K, dlm1)
        x_tilde1 = tf.reshape(x_tilde1, [-1, int(in_feat)])  # (B*K, dlm1)
        labels = tf.reshape(labels, [-1, int(in_feat)])  # (B*K, dlm1)
        d_labels = tf.reshape(tf.reduce_sum(labels * labels, 1), [-1, self.config.min_nn])  # (B*N, d)

        name_flayer = name + "_flayer0"
        labels = tf.nn.leaky_relu(tf.matmul(labels, self.W[name_flayer]) + self.b[name_flayer])  # (B*K, F)
        name_flayer = name + "_flayer1"
        labels_exp = tf.expand_dims(labels, 1)  # (B*K, 1, F)
        labels1 = labels_exp + 0.0
        for ss in range(1, int(in_feat / stride_th1)):
            labels1 = tf.concat([labels1, self.myroll(labels_exp, shift=(ss + 1) * stride_th1, axis=2)],axis=1)  # (B*K, dlm1/stride, dlm1)
        labels2 = labels_exp + 0.0
        for ss in range(1, int(out_feat / stride_th2)):
            labels2 = tf.concat([labels2, self.myroll(labels_exp, shift=(ss + 1) * stride_th2, axis=2)],axis=1)  # (B*K, dl/stride, dlm1)
        
        theta1 = tf.matmul(tf.reshape(labels1, [-1, int(in_feat)]),self.W[name_flayer + "_th1"])  # (B*K*dlm1/stride, R*stride)
        theta1 = tf.reshape(theta1, [-1, self.config.rank_theta, int(in_feat)]) + self.b[name_flayer + "_th1"]
        theta2 = tf.matmul(tf.reshape(labels2, [-1, int(in_feat)]),self.W[name_flayer + "_th2"])  # (B*K*dl/stride, R*stride)
        theta2 = tf.reshape(theta2, [-1, self.config.rank_theta, int(out_feat)]) + self.b[name_flayer + "_th2"]
        thetal = tf.expand_dims(tf.matmul(labels, self.W[name_flayer + "_thl"]) + self.b[name_flayer + "_thl"],2)  # (B*K, R, 1)

        x = tf.matmul(theta1, tf.expand_dims(x_tilde1, 2))  # (B*K, R, 1)
        x = tf.multiply(x, thetal)  # (B*K, R, 1)
        x = tf.matmul(theta2, x, transpose_a=True)[:, :, 0]  # (B*K, dl)

        x = tf.reshape(x, [-1, self.config.min_nn, int(out_feat)])  # (N, k, dl)
        x = tf.multiply(x, tf.expand_dims(tf.exp(-tf.div(d_labels, 10)), 2))  # (N, d, dl)
        x = tf.reduce_sum(x, 1)  # (N, dl) *era reduce_mean


        # divide the output in patch to be compatible with the other computations
        #x = tf.reshape(x, [-1, self.N, int(out_feat)])  # (B, N, dl)

        if return_graph:
            return x, D
        else:
            return x

    def gconv_conv_inner2(self, p):
        # each internal op: (k+1, F), all in parallel: (N,k+1, F) N=#tot point in pc k = #neighbors (1024)
        p = tf.cast(p, tf.float64)
        # find central pixel
        p_central = p[0,:]  # each internal op: (1, F), all in parallel: (N,F) N=#tot point in pc
        p_central=tf.expand_dims(p_central,0)
        # distances between central pixels and all other pixels
        p=tf.expand_dims(p,0)
        central_norm = tf.reduce_sum(p_central*p_central,1) # each internal op: (1,), all in parallel:(N,)
        all_norms = tf.reduce_sum(p*p, 2) # each internal op: (k+1,F), all in parallel: (N,k+1)
        #D = tf.abs( tf.expand_dims(central_norm,1) + all_norms - 2*tf.matmul(p, tf.expand_dims(p_central,2)) [:,0] ) # (N,W)
        D = tf.abs( tf.expand_dims(central_norm,1) + all_norms - 2*tf.transpose(tf.squeeze(tf.matmul(p, tf.expand_dims(p_central,2)), 0)))
        p = tf.cast(p, tf.float32) 
        D = tf.cast(D, tf.float32)
        return D

    def lnl_aggregation(self, h_l, h_nl, b):

        return tf.div(h_l + h_nl, self.config.min_nn+1) + b

    # return h_l + b
    def batch_norm_wrapper(self, inputs, name, decay=0.999):

        def bn_train():
            if len(inputs.get_shape()) == 4:
                # for convolutional activations of size (batch, height, width, depth)
                batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            if len(inputs.get_shape()) == 3:
                # for activations of size (batch, points, features)
                batch_mean, batch_var = tf.nn.moments(inputs, [0, 1])
            if len(inputs.get_shape()) == 2:
                # for fully connected activations of size (batch, features)
                batch_mean, batch_var = tf.nn.moments(inputs, [0])
            train_mean = tf.assign(self.pop_mean['bn_pop_mean_' + name],self.pop_mean['bn_pop_mean_' + name] * decay + batch_mean * (1 - decay))

            train_var = tf.assign(self.pop_var['bn_pop_var_' + name], self.pop_var['bn_pop_var_' + name] * decay + batch_var * (1 - decay))

            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, self.beta['bn_beta_' + name],self.scale['bn_scale_' + name], 1e-3)


        def bn_test():
            return tf.nn.batch_normalization(inputs, self.pop_mean['bn_pop_mean_' + name],self.pop_var['bn_pop_var_' + name], self.beta['bn_beta_' + name],self.scale['bn_scale_' + name], 1e-3)

        normalized = tf.cond(self.is_training, bn_train, bn_test)
        return normalized


    def __make_compute_graph(self):

        def noise_extract(h):       
            # pre
            n_div = 20
            name_block = "pre"
            for i in range (self.config.pre_n_layers):
                h = tf.nn.conv1d(h, self.W[name_block+"_"+str(i) ], stride=1, padding="VALID")
                h = self.batch_norm_wrapper(h, name_block + str(i))
                h = tf.nn.leaky_relu(h)                
            #  prox
            name_block = "residual"
            for i in range(self.config.n_block): #self.config.n_block
                h_hold = h
                for j in range(self.config.conv_n_layers):#self.config.conv_n_layers
                    h_r = tf.reshape(h, [-1, self.config.Nfeat])
                    if j == 0:
                        p = self.number_points/n_div
                        for k in range(n_div): 
                            #partial_h=h_r[k*p:k*p+p, :]
                            partial_knn_matrix = self.knn_matrix[k*p:k*p+p, :]
                            partial_points = tf.gather(h_r, partial_knn_matrix)
                            partial_h_nl, partial_D = self.gconv_conv(partial_points, name_block + str(i) + "_nl_" + str(j), self.config.Nfeat,self.config.Nfeat, self.config.stride, self.config.stride,compute_graph=True, return_graph=True)
                            if k == 0:
                                h_nl = partial_h_nl +0.0
                                D = partial_D +0.0
                            else:
                                h_nl = tf.concat([h_nl, partial_h_nl], 0)
                                D = tf.concat([D, partial_D], 0)
                    else:
                        p = self.number_points/n_div
                        for k in range(n_div):
                            #partial_h=h_r[k*p:k*p+p, :]
                            partial_knn_matrix = self.knn_matrix[k*p:k*p+p, :]
                            partial_D = D[k*p:k*p+p,:]
                            partial_points = tf.gather(h_r, partial_knn_matrix)
                            partial_h_nl = self.gconv_conv(partial_points, name_block + str(i) + "_nl_" + str(j), self.config.Nfeat,self.config.Nfeat, self.config.stride, self.config.stride,compute_graph=False, return_graph=False, D= partial_D)               
                            if k == 0:
                                h_nl = partial_h_nl +0.0
                            else:
                                h_nl = tf.concat([h_nl, partial_h_nl], 0)
                    h_nl = tf.reshape(h_nl, [-1, self.N, self.config.Nfeat])
                    h_sl = tf.nn.conv1d(h, self.W[name_block + "_sl_" + str(i)+ "_" + str(j)], stride=1, padding="VALID")
                    h = self.lnl_aggregation(h_sl, h_nl, self.b[name_block + str(i) + "_" + str(j)])
                    h = self.batch_norm_wrapper(h, name_block + str(i) + "_" + str(j))
                    h = tf.nn.leaky_relu(h)
                h = h_hold + h
            

            # last - return to the space of points from the feature space
            h_r = tf.reshape(h, [-1, self.config.Nfeat]) #(N, 3) N: #tot of point in the pc
            name_block = "last"
            p = self.number_points/n_div
            for k in range(n_div):
                #partial_h=h_r[k*p:k*p+p, :]
                partial_knn_matrix = self.knn_matrix[k*p:k*p+p, :]
                partial_points = tf.gather(h_r, partial_knn_matrix)
                partial_h_nl = self.gconv_conv(partial_points, name_block +  "_nl_0", self.config.Nfeat,self.config.input_ch, self.config.stride, self.config.stride,compute_graph=True, return_graph=False)              
                if k == 0:
                    h_nl = partial_h_nl +0.0
                else:
                    h_nl = tf.concat([h_nl, partial_h_nl], 0)
            #h_nl, D= self.gconv_conv(points, name_block + "_nl_0", self.config.Nfeat, self.config.input_ch, self.config.stride,self.config.stride, compute_graph=True, return_graph=True)
            h_nl = tf.reshape(h_nl, [-1, self.N, self.config.input_ch])
            h_sl = tf.nn.conv1d(h, self.W[name_block + "_sl_0"], stride=1, padding="VALID")
            h = self.lnl_aggregation(h_sl, h_nl, self.b[name_block + "_0"])
            return h
       
        self.n_hat = noise_extract(self.x_noisy)
        self.x_hat = self.x_noisy - self.n_hat

    def denoise(self, data_noisy, knn_matrix, number_points):
        feed_dict = {self.x_noisy: data_noisy, self.number_points: number_points,self.is_training: False, self.knn_matrix: knn_matrix}

        denoised_batch = self.sess.run(self.x_hat, feed_dict = feed_dict)
        return denoised_batch

    def do_variables_init(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    def save_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)


