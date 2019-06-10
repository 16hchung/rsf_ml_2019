import pandas as pd
from math import *
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

def pseudo_rsfs_data(X,y,hdrs):
    n, d = X.shape
    pseudo_X = pd.DataFrame()
    for hdr in hdrs:
        liq_mean = X[hdr].loc[y==1].mean()
        ice_mean = X[hdr].loc[y==0].mean()
        liq_var  = X[hdr].loc[y==1].var()
        ice_var  = X[hdr].loc[y==0].var()
        pseudo_ice = np.random.normal(loc=ice_mean, scale=sqrt(ice_var), size=int(n/2))
        pseudo_liq = np.random.normal(loc=liq_mean, scale=sqrt(liq_var), size=int(n/2))
        pseudo_X[hdr] = np.concatenate((pseudo_ice, pseudo_liq),axis=None)
    return pseudo_X

def pseudo_rsfs_from_rdf(rdf_fname, cart_fname, sigma, n=768, save_fname=""):
    rdfs = pd.read_csv(rdf_fname, skiprows=4, names=["mu", "g"], delim_whitespace=True, usecols=[1,2])
    sim_vol = 1
    natoms = 0
    cartf = open(cart_fname)
    for i, line in enumerate(cartf):
        if i==3:
            natoms = int(line)
        elif i>4 and i<8:
            dim = [float(x) for x in line.split()]
            sim_vol *= dim[1] - dim[0]
        elif i>8:
            break
    density = natoms/sim_vol
    
    mean_rsfs = rdfs.copy()
    omega = 4/3*np.pi*((rdfs["mu"] + 1.25 * sigma)**3 - (rdfs["mu"] - 1.25 * sigma)**3)
    mean_rsfs["g"] = mean_rsfs["g"] * density * omega
    pseudo_X = pd.DataFrame()
    for i, row in mean_rsfs.iterrows():
        ftr_name = "mu%.2f" % (row["mu"] - .05)
        pseudo = np.random.normal(loc=row["g"], scale=sigma, size=n)
        pseudo_X[ftr_name] = pseudo
    return pseudo_X

if __name__ == "__main__":
    rdf_fname = "ice.rdf"
    cart_fname = "ice_dump_250K_10000.dat"
    pseudo_rsfs_from_rdf(rdf_fname, cart_fname, .02)
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--plot_hist", action="store_true")
    #opts = parser.parse_args()




