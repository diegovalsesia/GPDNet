import numpy as np
import point_cloud_utils as pcu
import scipy.io as sio

def compute_C2C(a, b, name =[], save_var = False):

	# dists_a_to_b : matrix [a.shape[0],1] contains the shortest squared distance between each point in a and the points in b
	# corrs_a_to_b is of shape (a.shape[0],) and contains the index into b of the closest point for each point in a

    dists_a_to_b, corrs_a_to_b = pcu.point_cloud_distance(a, b)

    dists_b_to_a, corrs_b_to_a = pcu.point_cloud_distance(b, a)
    if save_var:
        sio.savemat(name + '_'+'C2C_var.mat',{'dist_a_b': dists_a_to_b, 'corr_a_b':corrs_a_to_b,'dist_b_a': dists_b_to_a, 'corr_b_a':corrs_b_to_a})

    #sum1= np.sum(np.sqrt(dists_a_to_b))
    sum1= np.sum(dists_a_to_b)
    #sum2= np.sum(np.sqrt(dists_b_to_a))
    sum2= np.sum(dists_b_to_a)    
    add1= sum1/(2*a.shape[0])
    
    add2= sum2/(2*b.shape[0])

    c2c=add1+add2
    
    return c2c