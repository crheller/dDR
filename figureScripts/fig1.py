"""
Averbeck & Lee type cartoon to help explain the rationale behind dDR, and what d'^2 is
"""
from dDR.utils.decoding import compute_dprime
from dDR.utils.plotting import compute_ellipse

from scipy.optimize import curve_fit
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

# TODO: Add decoding axes, edit in inkscape, add figure / text to mansucript

np.random.seed(123)

savefig = True
fig_name = os.path.join(os.getcwd(), 'figures/fig1.svg')

nsamp = 50000
amp = 2.5

def gaus(x, u, s, a):
    return a * np.exp( - ( ((x-u)**2) / (2*s**2) ) ) 

# figure layout
f = plt.figure(figsize=(6, 4))
s1 = plt.subplot2grid((6, 3), (0, 0), rowspan=3, colspan=1)
s2 = plt.subplot2grid((6, 3), (0, 1), rowspan=3, colspan=1)
s3 = plt.subplot2grid((6, 3), (0, 2), rowspan=3, colspan=1)
h11 = plt.subplot2grid((6, 3), (4, 0), rowspan=1, colspan=1)
h12 = plt.subplot2grid((6, 3), (5, 0), rowspan=1, colspan=1)
h21 = plt.subplot2grid((6, 3), (4, 1), rowspan=1, colspan=1)
h22 = plt.subplot2grid((6, 3), (5, 1), rowspan=1, colspan=1)
h31 = plt.subplot2grid((6, 3), (4, 2), rowspan=1, colspan=1)
h32 = plt.subplot2grid((6, 3), (5, 2), rowspan=1, colspan=1)

# generate three sets of pseudo-data for two example neurons

# EXAMPLE 1
u1 = [1, 1]
u2 = [2, 2]
cov = np.array([
    [1, np.sqrt(2) / 2],
    [np.sqrt(2) / 2, 1]
])
x1 = np.random.multivariate_normal(u1, cov, nsamp).T
x2 = np.random.multivariate_normal(u2, cov, nsamp).T
e1 = compute_ellipse(x1[0], x1[1])
e2 = compute_ellipse(x2[0], x2[1])
s1.plot(e1[0], e1[1])
s1.plot(e2[0], e2[1])
# get decoding axis / plot histograms
d = compute_dprime(x1, x2)
wopt = d.wopt / np.linalg.norm(d.wopt)
s1.plot([1.5, 1.5+amp*wopt[0][0]], [1.5, 1.5+amp*wopt[1][0]], "k-")
s1.plot([1.5, 1.5-amp*wopt[0][0]], [1.5, 1.5-amp*wopt[1][0]], "k-")
s1.plot([1.5, 1.5+amp*-wopt[1][0]], [1.5, 1.5+(amp*wopt[0][0])], "k--")
s1.plot([1.5, 1.5-amp*-wopt[1][0]], [1.5, 1.5-(amp*wopt[0][0])], "k--")
y = x1.T.dot(wopt).squeeze()

y, x = np.histogram(y)
f1opt, f1cov = curve_fit(gaus, x[1:], y)
f1opt[-1] = 1
h11.plot(np.arange(x[0], x[-1], 0.01), gaus(np.arange(x[0], x[-1], 0.01), *f1opt))
y = x2.T.dot(wopt).squeeze()

y, x = np.histogram(y)
f1opt, f1cov = curve_fit(gaus, x[1:], y)
f1opt[-1] = 1
h11.plot(np.arange(x[0], x[-1], 0.01), gaus(np.arange(x[0], x[-1], 0.01), *f1opt))

d = compute_dprime(x1, x2, diag=True)
wopt = d.wopt / np.linalg.norm(d.wopt)
s1.plot([1.5, 1.5+amp*wopt[0][0]], [1.5, 1.5+amp*wopt[1][0]], "grey")
s1.plot([1.5, 1.5-amp*wopt[0][0]], [1.5, 1.5-amp*wopt[1][0]], "grey")
s1.plot([1.5, 1.5+amp*-wopt[1][0]], [1.5, 1.5+(amp*wopt[0][0])], "grey", linestyle="--")
s1.plot([1.5, 1.5-amp*-wopt[1][0]], [1.5, 1.5-(amp*wopt[0][0])], "grey", linestyle="--")
y = x1.T.dot(wopt).squeeze()

y, x = np.histogram(y)
f1opt, f1cov = curve_fit(gaus, x[1:], y)
f1opt[-1] = 1
h12.plot(np.arange(x[0], x[-1], 0.01), gaus(np.arange(x[0], x[-1], 0.01), *f1opt))
y = x2.T.dot(wopt).squeeze()

y, x = np.histogram(y)
f1opt, f1cov = curve_fit(gaus, x[1:], y)
f1opt[-1] = 1
h12.plot(np.arange(x[0], x[-1], 0.01), gaus(np.arange(x[0], x[-1], 0.01), *f1opt))


# EXAMPLE 2
u1 = [1, 1.5]
u2 = [2.41, 1.5]
cov = np.array([
    [1, np.sqrt(2) / 2],
    [np.sqrt(2) / 2, 1]
])
x1 = np.random.multivariate_normal(u1, cov, nsamp).T
x2 = np.random.multivariate_normal(u2, cov, nsamp).T
e1 = compute_ellipse(x1[0], x1[1])
e2 = compute_ellipse(x2[0], x2[1])
s2.plot(e1[0], e1[1])
s2.plot(e2[0], e2[1])
# get decoding axis / plot histograms
d = compute_dprime(x1, x2)
wopt = d.wopt / np.linalg.norm(d.wopt)
s2.plot([1.5, 1.5+amp*wopt[0][0]], [1.5, 1.5+amp*wopt[1][0]], "k-")
s2.plot([1.5, 1.5-amp*wopt[0][0]], [1.5, 1.5-amp*wopt[1][0]], "k-")
s2.plot([1.5, 1.5+amp*-wopt[1][0]], [1.5, 1.5+(amp*wopt[0][0])], "k--")
s2.plot([1.5, 1.5-amp*-wopt[1][0]], [1.5, 1.5-(amp*wopt[0][0])], "k--")
y = x1.T.dot(wopt).squeeze()

y, x = np.histogram(y)
f1opt, f1cov = curve_fit(gaus, x[1:], y)
f1opt[-1] = 1
h21.plot(np.arange(x[0], x[-1], 0.01), gaus(np.arange(x[0], x[-1], 0.01), *f1opt))
y = x2.T.dot(wopt).squeeze()

y, x = np.histogram(y)
f1opt, f1cov = curve_fit(gaus, x[1:], y)
f1opt[-1] = 1
h21.plot(np.arange(x[0], x[-1], 0.01), gaus(np.arange(x[0], x[-1], 0.01), *f1opt))

d = compute_dprime(x1, x2, diag=True)
wopt = d.wopt / np.linalg.norm(d.wopt)
s2.plot([1.5, 1.5+amp*wopt[0][0]], [1.5, 1.5+amp*wopt[1][0]], "grey")
s2.plot([1.5, 1.5-amp*wopt[0][0]], [1.5, 1.5-amp*wopt[1][0]], "grey")
s2.plot([1.5, 1.5+amp*-wopt[1][0]], [1.5, 1.5+(amp*wopt[0][0])], "grey", linestyle="--")
s2.plot([1.5, 1.5-amp*-wopt[1][0]], [1.5, 1.5-(amp*wopt[0][0])], "grey", linestyle="--")
y = x1.T.dot(wopt).squeeze()

y, x = np.histogram(y)
f1opt, f1cov = curve_fit(gaus, x[1:], y)
f1opt[-1] = 1
h22.plot(np.arange(x[0], x[-1], 0.01), gaus(np.arange(x[0], x[-1], 0.01), *f1opt))
y = x2.T.dot(wopt).squeeze()

y, x = np.histogram(y)
f1opt, f1cov = curve_fit(gaus, x[1:], y)
f1opt[-1] = 1
h22.plot(np.arange(x[0], x[-1], 0.01), gaus(np.arange(x[0], x[-1], 0.01), *f1opt))

# EXAMPLE 3
u1 = [2, 1]
u2 = [1, 2]
cov = np.array([
    [1, np.sqrt(2) / 2],
    [np.sqrt(2) / 2, 1]
])
x1 = np.random.multivariate_normal(u1, cov, nsamp).T
x2 = np.random.multivariate_normal(u2, cov, nsamp).T
e1 = compute_ellipse(x1[0], x1[1])
e2 = compute_ellipse(x2[0], x2[1])
s3.plot(e1[0], e1[1])
s3.plot(e2[0], e2[1])
# get decoding axis / plot histograms
d = compute_dprime(x1, x2)
wopt = d.wopt / np.linalg.norm(d.wopt)
s3.plot([1.5, 1.5+amp*wopt[0][0]], [1.5, 1.5+amp*wopt[1][0]], "k-")
s3.plot([1.5, 1.5-amp*wopt[0][0]], [1.5, 1.5-amp*wopt[1][0]], "k-")
s3.plot([1.5, 1.5+amp*-wopt[1][0]], [1.5, 1.5+(amp*wopt[0][0])], "k--")
s3.plot([1.5, 1.5-amp*-wopt[1][0]], [1.5, 1.5-(amp*wopt[0][0])], "k--")
y = x1.T.dot(wopt).squeeze()
y, x = np.histogram(y)
f1opt, f1cov = curve_fit(gaus, x[1:], y)
f1opt[-1] = 1
h31.plot(np.arange(x[0], x[-1], 0.01), gaus(np.arange(x[0], x[-1], 0.01), *f1opt))
y = x2.T.dot(wopt).squeeze()
y, x = np.histogram(y)
f1opt, f1cov = curve_fit(gaus, x[1:], y)
f1opt[-1] = 1
h31.plot(np.arange(x[0], x[-1], 0.01), gaus(np.arange(x[0], x[-1], 0.01), *f1opt))

d = compute_dprime(x1, x2, diag=True)
wopt = d.wopt / np.linalg.norm(d.wopt)
s3.plot([1.5, 1.5+amp*wopt[0][0]], [1.5, 1.5+amp*wopt[1][0]], "grey")
s3.plot([1.5, 1.5-amp*wopt[0][0]], [1.5, 1.5-amp*wopt[1][0]], "grey")
s3.plot([1.5, 1.5+amp*-wopt[1][0]], [1.5, 1.5+(amp*wopt[0][0])], "grey", linestyle="--")
s3.plot([1.5, 1.5-amp*-wopt[1][0]], [1.5, 1.5-(amp*wopt[0][0])], "grey", linestyle="--")
y = x1.T.dot(wopt).squeeze()
y, x = np.histogram(y)
f1opt, f1cov = curve_fit(gaus, x[1:], y)
f1opt[-1] = 1
h32.plot(np.arange(x[0], x[-1], 0.01), gaus(np.arange(x[0], x[-1], 0.01), *f1opt))
y = x2.T.dot(wopt).squeeze()
y, x = np.histogram(y)
f1opt, f1cov = curve_fit(gaus, x[1:], y)
f1opt[-1] = 1
h32.plot(np.arange(x[0], x[-1], 0.01), gaus(np.arange(x[0], x[-1], 0.01), *f1opt))

for a in [s1, s2, s3]:
    a.set_xlim((-1, 4))
    a.set_ylim((-1, 4))
for a in [s1, s2, s3, h11, h12, h21, h22, h31, h32]:
    a.set_xticks([])
    a.set_yticks([])

f.tight_layout()

if savefig: 
    f.savefig(fig_name)

plt.show()