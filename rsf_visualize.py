import datetime
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import manifold
import pdb
import rsf_load_data

def plot_rsf_histograms(rsf_df, FTR_HDRS, scale=True, trim=True, log_y=True, nbins=100, dir_suffix="", debug=False, verbose=True):
    HIST_DIR = "rsf_histograms_" + dir_suffix + "/"
    if not os.path.exists(HIST_DIR):
        os.mkdir(HIST_DIR)
        print("Directory " , HIST_DIR ,  " Created ")

    liq_scaler = None
    sol_scaler = None
    if scale:
        scaler, rsf_df[FTR_HDRS] = rsf_load_data.scale_data(rsf_df[FTR_HDRS])
    for hdr in FTR_HDRS:
        if debug:
            pdb.set_trace()
        fpath_scaled = HIST_DIR + "scaled_hist" + hdr + ".png"
        fpath        = fpath_scaled if scale else HIST_DIR + "hist" + hdr + ".png"
        minrsf = rsf_df[hdr].min()
        maxrsf = rsf_df[hdr].max()
        if minrsf == 0 and maxrsf == 0: #if all at this mu are 0 then skip
            if verbose:
                print("skipping ", hdr)
            continue
        if trim:
            minrsf = -2
            maxrsf = 2
        bins = np.linspace(minrsf, maxrsf, nbins)
        plt.hist(rsf_df[hdr].loc[rsf_df["y"]==0], bins=bins, log=log_y,alpha=0.5, label='ice', edgecolor='k')
        plt.hist(rsf_df[hdr].loc[rsf_df["y"]==1], bins=bins, log=log_y,alpha=0.5, label='liq', edgecolor='k')
        plt.legend(loc='upper right')
        plt.savefig(fpath)
        if verbose:
            print("finished plotting histogram for ", hdr)
        plt.clf()

    return

def plot_rsf_tsne(rsf_df, FTR_HDRS, fname=""):
    TSNE_DIR = "rsf_tsne_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") + "/"
    if not os.path.exists(TSNE_DIR):
        os.mkdir(TSNE_DIR)
        print("Directory " , TSNE_DIR ,  " Created ")

    #for p in range(10, 200, 10):
    # run tsne on rsf matrix
    X_transform = manifold.TSNE(n_components=2).fit_transform(rsf_df[FTR_HDRS])
    X_transform_mix = X_transform[rsf_df["y"]==2]
    X_transform_liq = X_transform[rsf_df["y"]==1]
    X_transform_ice = X_transform[rsf_df["y"]==0]
    # plot
    fname = TSNE_DIR + "tsne_perplex.png"# + str(p) + ".png"
    plt.scatter(X_transform_mix[:,0], X_transform_mix[:,1], alpha=.5, label="mix", edgecolor="k")
    plt.scatter(X_transform_liq[:,0], X_transform_liq[:,1], alpha=.5, label="liq", edgecolor="k")
    plt.scatter(X_transform_ice[:,0], X_transform_ice[:,1], alpha=.5, label="ice", edgecolor="k")
    plt.legend(loc='upper right')
    plt.savefig(fname)
    plt.clf()
    #print("plotted for perplexity ", p)
    return

def plot_rsf_means(rsf_df, FTR_HDRS, fname="means_of_rsfs_minmax_errbars1.png"):
    pdb.set_trace()
    liq_means = rsf_df[rsf_df["y"] == 1].mean()
    #liq_stds  = rsf_df[rsf_df["y"] == 1].std()
    liq_mins  = liq_means - rsf_df[rsf_df["y"] == 1].min()
    liq_maxs  = rsf_df[rsf_df["y"] == 1].max() - liq_means
    liq_error = np.array([liq_mins, liq_maxs])
    
    ice_means = rsf_df[rsf_df["y"] == 0].mean()
    ice_mins  = ice_means - rsf_df[rsf_df["y"] == 0].min()
    ice_maxs  = rsf_df[rsf_df["y"] == 0].max() - ice_means
    #ice_stds  = rsf_df[rsf_df["y"] == 0].std()
    ice_error = np.array([ice_mins, ice_maxs])
    
    mus = [float(s[2:]) for s in FTR_HDRS]
    plt.errorbar(mus, liq_means[FTR_HDRS], liq_error[:, :-1], capsize=3, label="liq")
    plt.errorbar(mus, ice_means[FTR_HDRS], ice_error[:, :-1], capsize=3, label="ice")
    #plt.errorbar(mus, liq_means[FTR_HDRS], liq_std[FTR_HDRS], capsize=3, label="liq")
    #plt.errorbar(mus, ice_means[FTR_HDRS], ice_std[FTR_HDRS], capsize=3, label="ice")
    plt.legend(loc="upper right")
    plt.savefig(fname)
    plt.clf()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--liq_rsfs", type=str, default="")
    parser.add_argument("--ice_rsfs", type=str, default="")
    parser.add_argument("--plot_hist", action="store_true")
    parser.add_argument("--dir_suffix", default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nbins", type=int, default=100)
    parser.add_argument("--no_log_y", action="store_false")
    parser.add_argument("--not_verbose", action="store_false")
    parser.add_argument("--plot_tsne", action="store_true")
    parser.add_argument("--plot_rsf_means", action="store_true")
    parser.add_argument("--tsne_fname", type=str, default="")
    parser.add_argument("--tsne_fake_rsfs", action="store_true")
    parser.add_argument("--tsne_fake_fromrdf", action="store_true")
    opts = parser.parse_args()

    rsf_df, FTR_HDRS = (rsf_load_data.load_rsfs_df(liq_fname=opts.liq_rsfs, sol_fname=opts.ice_rsfs)
            if opts.liq_rsfs and opts.ice_rsfs
            else rsf_load_data.load_rsfs_df()
            )
    if opts.plot_hist:
        plot_rsf_histograms(rsf_df, FTR_HDRS, scale=True, trim=False, log_y=opts.no_log_y, nbins=opts.nbins,
                            dir_suffix=opts.dir_suffix, debug=opts.debug,
                            verbose=opts.not_verbose)
    if opts.plot_tsne:
        if opts.tsne_fake_rsfs:
            pseudo_X = rsf_load_data.pseudo_rsfs_data(rsf_df[FTR_HDRS], rsf_df["y"], FTR_HDRS)
            pseudo_X["y"] = rsf_df["y"]
            rsf_df = pseudo_X
        elif opts.tsne_fake_fromrdf:
            mixed_rdf_fname = "mixed.rdf"
            mixed_cart_fname = "dump_250K_1960000.dat"
            n_rsfs_to_generate = 768
            mixed_pseudo = rsf_load_data.pseudo_rsfs_from_rdf(mixed_rdf_fname, mixed_cart_fname, .02, n=n_rsfs_to_generate, type_dist="exp")
            mixed_pseudo["y"] = 2
            rsf_df = pd.concat([rsf_df, mixed_pseudo], ignore_index=True)
        plot_rsf_tsne(rsf_df, FTR_HDRS, fname=opts.tsne_fname)

    if opts.plot_rsf_means:
        plot_rsf_means(rsf_df, FTR_HDRS)
    return

if __name__ == "__main__":
    main()
