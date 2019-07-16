from math import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import model_selection
from sklearn import preprocessing

from .load_data import RSFDataLoader

class SVMRunner():
    def __init__(self, X_train, y_train, whiten_data=True, C=10):
        self.X_train = X_train
        self.y_train = y_train
        self.whiten_data = whiten_data
        self.C = C
        self.scaler = None
        
        if whiten_data:
            self.scaler, X_train = RSFDataLoader.scale_data(X_train)
        
        self.clf = svm.linearSVC() if not self.C else svm.LinearSVC(C=self.C)
        self.clf.fit(X_train, self.y_train)

    def validate_simple(self, X_val, y_val=None, message=None):
        if self.whiten_data:
            X_val = self.scaler.transform(X_val)
            
        val_ypred = self.clf.predict(X_val)

        print("Linear SVM with C={}".format(self.C))
        if message:
            print('\t' + message)
        if not y_val:
            print("\t{}% positive predictions".format(np.sum(val_ypred)*100 / len(val_ypred)))
        else:
            err = np.sum(np.abs(val_ypred - y_val)) / X_val.shape[0]
            print("\terror was {}%".format(err*100))
            
    def plot_confidence(self, plot_train=False, X_val=[], y_val=[], trim_std=False, nbins=100, fname=None, title="Decision Function"):
        plt.clf()
        import pdb;pdb.set_trace()
        maxconf = 0; minconf = 0
        if len(X_val):
            conf_val, maxconf, minconf = self._calc_confidence(X_val, trim_std)
        if plot_train:
            conf_train, maxtrain, mintrain = self._calc_confidence(self.X_train, trim_std)
            if not maxconf:
                maxconf = maxtrain
                minconf = mintrain
        bins = np.linspace(minconf, maxconf, nbins)
        
        if plot_train:
            self._plot_calculated_confidence(bins, conf_train, self.y_train, 'train')
        if len(X_val):
            self._plot_calculated_confidence(bins, conf_val, y_val, 'mixed')
            
        plt.title(title)
        plt.legend(loc="best")
        if fname:
            plt.savefig(fname)
        else:
            plt.show()
    
    def _calc_confidence(self, X, trim):
        if self.whiten_data:
            X = self.scaler.transform(X)
        confidence = self.clf.decision_function(X)
        if trim:
            mean = confidence.mean()
            maxconf = mean + confidence.std() * trim
            minconf = mean - confidence.std() * trim
        else:
            maxconf = confidence.max()
            minconf = confidence.min()
        return confidence, maxconf, minconf
    
    def _plot_calculated_confidence(self, bins, confidence, y=[], label_prefix=''):
        if len(y):
            plt.hist(confidence[y==RSFDataLoader.LIQ_LBL], bins=bins, density=True, 
                     alpha=.5, label='{}_liq'.format(label_prefix), edgecolor='k')
            plt.hist(confidence[y==RSFDataLoader.ICE_LBL], bins=bins, density=True, 
                     alpha=.5, label='{}_ice'.format(label_prefix), edgecolor='k')
        else:
            plt.hist(confidence, bins=bins, density=True, alpha=.5, label=label_prefix, edgecolor='k')
            

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

