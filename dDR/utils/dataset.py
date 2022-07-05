"""
Helper for manipulating real spike data
"""
import pickle
import os 
import dDR
ROOT_DIR = os.path.dirname(os.path.dirname(dDR.__file__))
datapath = os.path.join(ROOT_DIR, 'data/')

class Dataset:

    def __init__(self, spikeData=None, meta=None):
        """
        spikeData is dictionary of spike counts. For each snr/center frequency: matrix of reps x neuron x time
        meta is a dictionary of useful information about the recording / stim on/off bins
        """
        if spikeData is not None:
            self.spikeData = spikeData
            self.snrs = list(spikeData.keys())
            self.cfs = list(spikeData[self.snrs[0]].keys())
        if meta is not None:
            self.meta = meta

    def save(self, path=datapath, name=None):
        '''
        pickle Dataset object to datapath location with name "name".
        '''
        fh = open(f"{datapath}{name}.pickle", "wb")
        pickle.dump(self, fh)
        fh.close()

    def load(self, path=datapath, name='CRD002a'):
        '''
        Load example dataset. Datsets are named: CRD002a, CRD003b, CRD004a
        '''
        fh = open(f"{datapath}{name}.pickle", "rb")
        d = pickle.load(fh)
        return d
    

# helper for extracting data
def extract_data(snr=None, cf=None):
    """ 
    Helper to pull out the spike matrix for a certain set of snrs and/or center frequencies.

    Returns a matrix of snr X cf x repetition x neuron x time

    WIP - gets weird bc for CRD003a there are less reps of a couple of stim
    """
    if type(snrs) is str:
        snr_dim = 1
    else:
        snr_dim = len(snr)
    if type(cf) is int:
        cf_dim = 1
    else:
        cf_dim = len(cf)

    #spike_matrix = np.zeros((snr_dim, cf_dim, )