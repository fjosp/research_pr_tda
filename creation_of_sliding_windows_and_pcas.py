import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates

import pickle

import csv
import numpy as np
import datetime

# %matplotlib inline

#~ from ripser import ripser, plot_dgms

# %matplotlib notebook
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

import scipy.interpolate as interp

import ipywidgets as widgets
from IPython.display import display

import warnings
warnings.filterwarnings('ignore')


# This code was taken from github TDAlabs

def getSlidingWindow(x, dim, Tau, dT):
    '''
    This function takes time series x (without time-part)
    and returns a massive X, which has sliding windows as columns
    '''
    N = len(x)
    NWindows = int(np.floor((N-dim*Tau)/dT)) # The number of windows
    if NWindows <= 0:
        print("Error: Tau too large for signal extent")
        return np.zeros((3, dim))
    X = np.zeros((NWindows, dim)) # Create a 2D array which will store all windows
    idx = np.arange(N)
    for i in range(NWindows):
        # Figure out the indices of the samples in this window
        idxx = dT*i + Tau*np.arange(dim) 
        start = int(np.floor(idxx[0]))
        end = int(np.ceil(idxx[-1]))+2
        if end >= len(x):
            X = X[0:i, :]
            break
        # Do spline interpolation to fill in this window, and place
        # it in the resulting array
        X[i, :] = interp.spline(idx[start:end+1], x[start:end+1], idxx)
    return X
    
whole_data = np.genfromtxt('whole_dataset.csv', delimiter=',')


pkl_file = open('times.pkl', 'rb')
times = pickle.load(pkl_file)
pkl_file.close()

with open('titles.csv', 'r') as f:
    reader = csv.reader(f)
    titles = list(reader)[0]

def sl_w_pca_2(x, dim=50, Tau=1, dT=1):
    '''
    This function takes ts and returns massive of sliding windows 
    for each of them and massive of two most important components of it
    '''
    X = getSlidingWindow(x, dim, Tau, dT)
    pca = PCA(n_components = 2)
    Y = pca.fit_transform(X)
    return (X,Y)

sliding_windows = [] #dataset with sliding windows massives for each card_id
pca_transforms = [] #dataset with pca(2) transforms of sliding windows
counter = 0
for time_series in whole_data:
    #~ if counter == 3000:
        #~ break
    w, p = sl_w_pca_2(time_series, dim=50, Tau=1, dT=1)
    sliding_windows.append(w)
    pca_transforms.append(p)
    #~ print(counter)
    counter +=1
    
output = open('sliding_windows.pkl', 'wb')
pickle.dump(sliding_windows, output)
output.close()



output = open('pca_transforms.pkl', 'wb')
pickle.dump(pca_transforms, output)
output.close()
