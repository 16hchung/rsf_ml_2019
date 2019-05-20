import pandas as pd
import numpy as np
from sklearn import preprocessing
import pdb

def scale_data(X):
    scaler = preprocessing.StandardScaler().fit(X)
    return scaler, scaler.transform(X)

def load_rsfs_df(liq_fname="rsf_data_liq_10000.dat", sol_fname="rsf_data_ice_10000.dat"):
    # constants
    NROWS_SKIP = 1

    Y_HDR = "y"

    dmu = 0.10 # rsf Gaussian-mean step [A].
    mu = np.arange(0.0,6.0+dmu,dmu) # rsf Gaussian mean [A].
    FTR_HDRS = ["mu%.2f" % number for number in mu]

    # load data 
    df_liq = pd.read_csv(liq_fname, skiprows=NROWS_SKIP, names=FTR_HDRS, delim_whitespace=True)
    df_ice = pd.read_csv(sol_fname, skiprows=NROWS_SKIP, names=FTR_HDRS, delim_whitespace=True)

    # label = 1 if molec is in liquid phase
    df_liq[Y_HDR] = 1
    df_ice[Y_HDR] = 0
    
    # form one data set with both labels
    df = pd.concat([df_ice, df_liq], ignore_index=True)
 
    return df, FTR_HDRS


