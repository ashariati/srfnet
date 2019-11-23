import numpy as np
import scipy # use numpy if scipy unavailable
import scipy.linalg # use numpy if scipy unavailable
import pdb
import warnings

def ransac(data, model, m, k, t):

    warnings.filterwarnings('error')

    iters = 0
    n_best = 0
    best_model = None
    best_err = None
    inlier_idxs = None
    while iters < k:

        sample_idxs, test_idxs = random_partition(m, data.shape[0])
        maybe_inliers = data[sample_idxs, :]
        test_points = data[test_idxs, :]

        maybe_model = model.fit(maybe_inliers)
        test_err = model.get_error(test_points, maybe_model)

        also_idxs = test_idxs[test_err < t]
        also_inliers = data[also_idxs, :]

        inliers = np.concatenate( (maybe_inliers, also_inliers) )
        n_inliers = inliers.shape[0]

        if n_inliers > n_best:

            best_model = model.fit(inliers)
            best_err = model.get_error(inliers, best_model)
            inlier_idxs = np.concatenate( (sample_idxs, also_idxs) )
            n_best = n_inliers

        iters += 1

    return best_model, {'inliers_idxs' : inlier_idxs, 'errors' : best_err}

def random_partition(n,n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange( n_data )
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

class LinearLeastSquaresModel:
    """linear system solved using linear least squares

    This class serves as an example that fulfills the model interface
    needed by the ransac() function.
    
    """
    def __init__(self,input_columns,output_columns,debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    def fit(self, data):
        A = np.vstack([data[:,i] for i in self.input_columns]).T
        B = np.vstack([data[:,i] for i in self.output_columns]).T
        x,resids,rank,s = scipy.linalg.lstsq(A,B)
        return x
    def get_error( self, data, model):
        A = np.vstack([data[:,i] for i in self.input_columns]).T
        B = np.vstack([data[:,i] for i in self.output_columns]).T
        B_fit = scipy.dot(A,model)
        err_per_point = np.sum((B-B_fit)**2,axis=1) # sum squared error per row
        return err_per_point

class EpipoleModel:

    def fit(self, field):

        lines = field_to_line(field)

        A = lines[:, :2]
        B = lines[:, -1]

        epipole, resids, rank, s = scipy.linalg.lstsq(A,B)

        return epipole

    def get_error(self, field, model):

        flow = field[:, :2]
        x = field[:, 2:]

        flow_mag = np.linalg.norm(flow, axis=1) 
        n_hat = flow / flow_mag[:, None]

        v = model - x
        v = v / np.linalg.norm(v, axis=1)[:, None]

        dp = np.sum(np.multiply(n_hat, v), axis=1)

        # flip if facing away
        nmask = dp < 0
        flipped_n = -n_hat[nmask, :]
        dp[nmask] = np.sum(np.multiply(flipped_n, v[nmask, :]), axis=1)

        # for numeric stability
        dp = np.clip(dp, -1, 1)

        # error
        dtheta = np.arccos(dp) * (180. / np.pi)

        return dtheta

def field_to_line(field):

    flow = field[:, :2]
    x = field[:, 2:]
    
    # normalize flow
    flow_mag = np.linalg.norm(flow, axis=1) 
    n_hat = flow / flow_mag[:, None]

    line = np.zeros(n_hat.shape)
    line[:, [0, 1]] = n_hat[:, [1, 0]]
    line[:, 1] = -line[:, 1] 
    
    C = np.sum(np.multiply(line, x), axis=1)

    return np.concatenate((line, C[:, None]), axis=1)
        
