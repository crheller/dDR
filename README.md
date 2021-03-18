# decoding-based Dimensionality Reduction (dDR)

dDR is a linear dimensionality reduction method for performing decoding analyses using neural population data in cases where the experimental data is limited. That is, cases where the number of neurons exceeds the number of observations (e.g. stimulus repetitions). dDR takes advantage of the fact that high-dimensional neural population data often exhibits low-dimensional, stimulus-independent dynamics. These low-dimensional activity patterns can be measured reliably, even when experimental repetitions are limited. Therefore, they can be leveraged to reliably measure decoding accuracy of both sensory stimuli and/or behavior. The process for performing dDR is explained graphically below. 

![alt text](figures/final/fig3.png "dDR procedure")

## Installing
#### Install editable version with iPython demos and dependencies:
```
git clone https://github.com/crheller/dDR.git
cd /path/to/dDR
pip install -e .[demos]
```
or:
```
git clone git@github.com:crheller/dDR.git
cd /path/to/dDR
pip install -e .[demos]
```
To leave off iPython notebook dependencies (`jupyter`), after cloning and switching into `dDR` directory, just use:
```
pip install -e .
```
#### Install base package only
Alternatively, the base package, without demos, can be installed directly from github:
```
pip install git+https://github.com/crheller/dDR.git
```

## Using dDR
For an introduction on using the dDR class for neural decoding, please see the included demo ipython notebook, `dDR_demo.ipynb`
