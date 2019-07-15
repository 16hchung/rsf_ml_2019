from math import *

import pandas as pd
import numpy as np
from sklearn import preprocessing

class RSFDataLoader:
    Y_HDR = 'y'
    LIQ_LBL = 1
    ICE_LBL = 0
    NO_LBL = 2
    NROWS_SKIP = 1
    _FTR_HDRS = None

    def __init__(self, rsfs_dir='rsfs/', rdfs_dir='rdfs/', cartesian_dir='cartesian/'):
        self.rsfs_dir = rsfs_dir
        self.rdfs_dir = rdfs_dir
        self.cart_dir = cartesian_dir

    @property
    def FTR_HDRS(self):
        if not self._FTR_HDRS:
            dmu = 0.10 # rsf Gaussian-mean step [A].
            mu = np.arange(0.0,6.0+dmu,dmu) # rsf Gaussian mean [A].
            self._FTR_HDRS = ["mu%.2f" % number for number in mu]
        return self._FTR_HDRS

    def scale_data(self, X):
        scaler = preprocessing.StandardScaler().fit(X)
        return scaler, scaler.transform(X)

    def load_rsfs(self, fname, label=None):
        fname = self.rsfs_dir + fname
        df = pd.read_csv(fname, skiprows=self.NROWS_SKIP, names=self.FTR_HDRS, delim_whitespace=True)
        
        y = None
        if label:
            y = pd.Series([label]*df.shape[0])
        return df, y

    def load_rsfs_liq_sol(self, liq_fname="rsf_data_liq_10000.dat", sol_fname="rsf_data_ice_10000.dat"):
        df_liq, liq_y = self.load_rsfs(liq_fname, self.LIQ_LBL)
        df_ice, ice_y = self.load_rsfs(sol_fname, self.ICE_LBL)
        
        df = pd.concat([df_ice, df_liq], ignore_index=True)
        y = pd.concat([ice_y, liq_y], ignore_index=True)
     
        return df, y

    def pseudo_rsfs_data(self, X, y, type_dist="gaus"):
        n, d = X.shape
        pseudo_X = pd.DataFrame()
        for hdr in hdrs:
            liq_mean = X[self.FTR_HDRS].loc[y==self.LIQ_LBL].mean()
            ice_mean = X[self.FTR_HDRS].loc[y==self.ICE_LBL].mean()
            liq_var  = X[self.FTR_HDRS].loc[y==self.LIQ_LBL].var()
            ice_var  = X[self.FTR_HDRS].loc[y==self.ICE_LBL].var()
            pseudo_ice = None
            pseudo_liq = None
            if type_dist == "gaus":
                pseudo_ice = np.random.normal(loc=ice_mean, scale=sqrt(ice_var), size=int(n/2))
                pseudo_liq = np.random.normal(loc=liq_mean, scale=sqrt(liq_var), size=int(n/2))
            elif type_dist == "exp":
                pseudo_ice = np.random.exponential(scale=ice_mean, size=int(n/2))
                pseudo_liq = np.random.exponential(scale=liq_mean, size=int(n/2))
            else:
                print("unsupported distribution type")
                return
            pseudo_X[hdr] = np.concatenate((pseudo_ice, pseudo_liq),axis=None)
        return pseudo_X

    def pseudo_rsfs_from_rdf(self, rdf_fname, cart_fname, sigma, type_dist="gaus", n=0, save_fname=""):
        rdf_fname = self.rdfs_dir + rdf_name
        save_fname = self.rsfs_dir + save_fname
        cart_fname = self.cart_dir + cart_fname
        
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
        if n<=0:
            n = natoms
        density = natoms/sim_vol
        
        mean_rsfs = rdfs.copy()
        omega = 4/3*np.pi*((rdfs["mu"] + 1.25 * sigma)**3 - (rdfs["mu"] - 1.25 * sigma)**3)
        mean_rsfs["g"] = mean_rsfs["g"] * density * omega
        pseudo_X = pd.DataFrame()
        for i, row in mean_rsfs.iterrows():
            ftr_name = "mu%.2f" % (row["mu"] - .05)
            pseudo = None
            if type_dist == "gaus":
                pseudo = np.random.normal(loc=row["g"], scale=sigma, size=n)
            elif type_dist == "exp":
                pseudo = np.random.exponential(scale=row["g"], size=n)
            else:
                print("unsupported distribution type")
                return
            pseudo_X[ftr_name] = pseudo
        return pseudo_X

'''
if __name__ == "__main__":
    rdf_fname = "ice.rdf"
    cart_fname = "ice_dump_250K_10000.dat"
    pseudo_rsfs_from_rdf(rdf_fname, cart_fname, .02)
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--plot_hist", action="store_true")
    #opts = parser.parse_args()
'''