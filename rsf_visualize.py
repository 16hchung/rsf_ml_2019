import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn import model_selection
import pdb
import rsf_load_data

def plot_rsf_histograms(scale=True, trim=True, log_y=True, nbins=100):
    HIST_DIR = "rsf_histograms/"

    rsf_df, FTR_HDRS = rsf_load_data.load_rsfs_df()
    liq_scaler = None
    sol_scaler = None
    if scale:
        scaler, rsf_df[FTR_HDRS] = rsf_load_data.scale_data(rsf_df[FTR_HDRS])
    for hdr in FTR_HDRS:
        fpath_scaled = HIST_DIR + "scaled_hist" + hdr + ".png"
        fpath        = fpath_scaled if scale else HIST_DIR + "hist" + hdr + ".png"
        minrsf = rsf_df[hdr].min()
        maxrsf = rsf_df[hdr].max()
        if minrsf == 0 and maxrsf == 0: #if all 
            continue
        if trim:
            minrsf = -2
            maxrsf = 2
        bins = np.linspace(minrsf, maxrsf, nbins)
        plt.hist(rsf_df[hdr].loc[rsf_df["y"]==0], bins=bins, log=log_y,alpha=0.5, label='ice', edgecolor='k')
        plt.hist(rsf_df[hdr].loc[rsf_df["y"]==1], bins=bins, log=log_y,alpha=0.5, label='liq', edgecolor='k')
        plt.legend(loc='upper right')
        plt.savefig(fpath)
        plt.clf()

    return

plot_rsf_histograms()
