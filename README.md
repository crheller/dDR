# decoding-based Dimensionality Reduction (dDR)

dDR is a linear dimensionality reduction method for performing decoding analyses using neural population data in cases where the experimental data is limited. That is, cases where the number of neurons exceeds the number of observations (e.g. stimulus repetitions). dDR takes advantage of the fact that high-dimensional neural population data often exhibits low-dimensional dynamics. These low-dimensional activity patterns can be measured reliably, even when experimental repetitions are limited, and can therefore be leveraged to reliably measure decoding accuracy of both sensory stimuli and/or behavior. The process for performing dDR is explained graphically below. 

![alt text](figures/final/fig3.png "dDR procedure")

## Installing
To install the package, first clone this repository using either `https` or `ssh` protocol. For `https`:
```
git clone https://github.com/crheller/dDR.git
```
or for `ssh`:
```
git clone git@github.com:crheller/dDR.git
```
Next, `cd` into the root `dDR` directory. For example: `cd <path-to-cloned-repo>/dDR`.

Finally, install the package using `pip`:
```
pip install -e .
```
In order to install the dependencies (`jupyter`) for using the demo ipython notebooks, make sure you install using:
```
pip install -e .[demos]
```

## Using dDR
For an introduction on using the dDR class for neural decoding, please see the included demo ipython notebook, `dDR_demo.ipynb`
