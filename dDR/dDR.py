import numpy as np

class dDR:

    def __init__(self, ddr2_init=None, n_additional_axes=None):
        '''
        Simple dimensionality reduction tool designed for performing neural decoding analysis. 
        
        ddr2_init: 
            If not None, a custom (for example) latent variable axis that you wish to use as your noise axis in the dDR decomposition.
            Must be a numpy array of shape N neurons x 1

        n_additional_axes:
            If not None, specifies the number of additional noise dimensions that should be included in the dDR composition. For example
            n_additional_axes=1 would correspond to a final dDR space with 3 dimensions (a signal axis + 2 noise dimensions).
        '''
        self.ddr2_init = ddr2_init
        self.n_additional_axes = n_additional_axes
         
    def fit(self, A, B):
        """
        Input:
            A and B are observation X neuron arrays.
                e.g. A could be stimulus 1 and B stimulus 2
        Return 
            None
        """
        # get dU
        dU = A.mean(axis=0, keepdims=True) - B.mean(axis=0, keepdims=True)
        dU = dU / np.linalg.norm(dU)

        # get first PC of mean centered data
        if self.ddr2_init is None:
            A0 = A - A.mean(axis=0, keepdims=True)
            B0 = B - B.mean(axis=0, keepdims=True)
            Xcenter = np.concatenate((A0, B0), axis=0)
            evals, evecs = np.linalg.eig(np.cov(Xcenter.T))
            evecs = evecs[:, np.argsort(evals)[::-1]]
            # in case cov is not full rank, which is possible in trial limited regime
            evecs = np.real(evecs)
            noise_axis = evecs[:, [0]].T
        else:
            noise_axis = self.ddr2_init
        
        self.noise_axis = noise_axis

        # figure out the axis that spans the plane with dU
        noise_on_dec = (np.dot(noise_axis, dU.T)) * dU
        orth_ax = noise_axis - noise_on_dec
        orth_ax /= np.linalg.norm(orth_ax)

        weights = np.concatenate((dU, orth_ax), axis=0)

        if self.n_additional_axes is not None:
            # remove dDR projection (deflate matrix)
            A0 = A - A.mean(axis=0, keepdims=True)
            B0 = B - B.mean(axis=0, keepdims=True)
            Xcenter = np.concatenate((A0, B0), axis=0)
            Xresidual = Xcenter - Xcenter.dot(weights.T).dot(weights)

            # find n additional axes, orthogonal to dDR plane
            evals, evecs = np.linalg.eig(np.cov(Xresidual.T))
            evecs = evecs[:, np.argsort(evals)[::-1]]
            evecs = np.real(evecs)
            noise_weights = evecs[:, :self.n_additional_axes].T
            weights = np.concatenate((weights, noise_weights), axis=0)

        self.components_ = weights


    def transform(self, X):
        """
        Projects X onto self.components_

        Input
            X: observation X neuron array
        Return 
            Xt: observation x nDim array, where nDim = 2 + self.n_additional_axes
        """
        if X.shape[-1] != self.components_.shape[-1]:
            raise ValueError("Dimension mismatch")
        return X.dot(self.components_.T)
    

    def fit_transform(self, A, B):
        """
        A and B are observation X neuron arrays.
            e.g. A could be stimulus 1 and B stimulus 2
        
        Return:
         transformed matrices. Each is an observation x dNim array, where
        nDim = 2 + self.n_additional_axes
        """
        self.fit(A, B)
        return self.transform(A), self.transform(B)
