import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn import model_selection
import pdb
import rsf_load_data
from math import *

def compile_rsfs_data():
    VAL_LIQ_FNAME = "rsf_data_liq_11000.dat"
    VAL_ICE_FNAME = "rsf_data_ice_11000.dat"

    Y_HDR = "y"

    df_train, FTR_HDRS = rsf_load_data.load_rsfs_df()
    df_val,   _        = rsf_load_data.load_rsfs_df(liq_fname=VAL_LIQ_FNAME, sol_fname=VAL_ICE_FNAME)

    # form one data set with both labels
    X_val   = df_val.loc[:, FTR_HDRS]
    X_train = df_train.loc[:, FTR_HDRS]
    y_val   = df_val[Y_HDR]
    y_train = df_train[Y_HDR]
    return X_train, y_train, X_val, y_val, FTR_HDRS

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

def scale_data(X):
    scaler = preprocessing.StandardScaler().fit(X)
    return scaler, scaler.transform(X)


def simple_SVM(X_train, y_train, X_val, y_val):
    clf = svm.LinearSVC(C=10)
    clf.fit(X_train, y_train)
    val_ypred = clf.predict(X_val)

    err = np.sum(np.abs(val_ypred - y_val)) / X_val.shape[0]
    print("Linear SVM:")# w/o normalization nor regularization:")
    print("    trained on pseudo, validated on 1st timestamp")
    print("    error was %d", err)

def plot_scores(train_scores, test_scores, title, xvals, fname, pltfxn, xlabel, ylabel="score"):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0.0, 1.1)
    plt.fill_between(xvals, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="r")
    plt.fill_between(xvals, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="g")
    pltfxn(xvals, train_scores_mean, 'o-', color="r",
           label="Training score")
    pltfxn(xvals, test_scores_mean, 'o-', color="g",
           label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig(fname)
    plt.clf()

def learning_curve(X, y, C=10, fname="linearsvm_learning.png"):
    cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    train_sizes, train_scores, test_scores = model_selection.learning_curve(
        svm.LinearSVC(C=C), X, y,
        cv=cv, scoring="accuracy", n_jobs=1)
    
    plot_scores(train_scores, test_scores, "Learning Curve with SVM", train_sizes, 
                fname, plt.plot, "training size")

def validation_curve(X, y, fname="linearsvm_validation.png"):
    param_range = [.01, .05, .1, .5, 1, 5, 10, 50, 100, 500]
    train_scores, test_scores = model_selection.validation_curve(
        svm.LinearSVC(), X, y, param_name="C", param_range=param_range,
        cv=5, scoring="accuracy", n_jobs=1)

    plot_scores(train_scores, test_scores, "Validation Curve with SVM", param_range, 
                fname, plt.semilogx, "C")

X_train, y_train, X_val, y_val, hdrs = compile_rsfs_data()
pdb.set_trace()
pseudo_X = pseudo_rsfs_data(X_train, y_train, hdrs)
#scaler, X_train = rsf_load_data.scale_data(X_train)
#pseudo_scaler, pseudo_X = rsf_load_data.scale_data(pseudo_X)
#validation_curve(pseudo_X, y_train, "pseudorsfs_linearsvm_validation.png")
#learning_curve(pseudo_X, y_train, fname="pseudorsfs_linearsvm_learning.png")

simple_SVM(pseudo_X, y_train, X_train, y_train)
#simple_SVM(pseudo_X, y_train, pseudo_scaler.transform(X_train), y_train)
#simple_SVM(X_train, y_train, scaler.transform(X_val), y_val)
