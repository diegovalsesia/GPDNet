class Config(object):

    def __init__(self):
        # directories
        self.save_dir = ''
        self.log_dir = ''
        self.train_data_file = ''
        self.save_txt = ''

        # input and layer params
        # input
        self.patch_size = [1024, 3]
        self.N = 1024
        self.input_ch = 3 

        # no. of layers
        self.pre_n_layers       = 3 
        self.pregconv_n_layers  = 0 
        #Net2
        self.n_block            = 2
        self.conv_n_layers      = 3
        # no. of features
        self.Nfeat              = 99 # must be multiple of 3
        self.pre_Nfeat          = [3,33,66,99]
        self.pre_fnet_Nfeat     = self.pre_Nfeat
        self.prox_fnet_Nfeat    = self.Nfeat

        # gconv params
        self.rank_theta         = 11
        self.stride             = self.Nfeat/3
        self.input_stride       = 1
        self.min_nn             = 16 


        # learning
        self.batch_size = 8
        self.batch_size_validation = 8
        self.grad_accum = 3
        self.N_iter = 50050
        self.starter_learning_rate = 1e-4
        self.end_learning_rate = 1e-5
        self.decay_step = 1000
        self.decay_rate = (self.end_learning_rate / self.starter_learning_rate) ** (float(self.decay_step) / self.N_iter)

        # debugging
        self.save_every_iter = 1000
        self.summaries_every_iter = 5
        self.validate_every_iter = 100
        self.test_every_iter = 1000
        self.print_image = 50000

        # testing
        self.knn = 975

        # noise std
        self.sigma = 0.02

