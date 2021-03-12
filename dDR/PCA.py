'''
Basic implementation of PCA using numpy eigendecomposition of covariance matrix
'''
import numpy as np

class PCA:

    def __init__(self, n_components=None, center=True):
        """
        n_components: int, number of components to return
        center: normalize data by mean centering each feature (neuron)
        """
        self.n_components = n_components
        self.center = center
         
    def fit(self, X):
        """
        Input:
            X is an Observation x Neuron numpy array
        Return 
            None
        """
        if self.n_components is None:
            n_components = X.shape[-1]
        else:
            n_components = self.n_components

        if self.center:
            # center the data by taking the mean across samples for each neuron
            X = X - X.mean(axis=0, keepdims=True)

        # covariance matrix
        cov = np.cov(X.T)

        # eigen-decomposition 
        evals, evecs = np.linalg.eig(cov)

        # sort 
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        # keep only real parts
        evals = np.real(evals)
        evecs = np.real(evecs)

        # save results to object
        self.components_ = evecs[:, :n_components].T
        self.explained_variance_ = evals[:n_components]
        self.explained_variance_ratio_ = evals[:n_components] / sum(evals)


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
    

    def fit_transform(self, X):
        """
        Input:
            X is observation x neuron numpy array
        
        Return:
            Transformed matrix. Each is an observation x self.n_components array
        """
        self.fit(X)
        return self.transform(X)