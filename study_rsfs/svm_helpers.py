import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import model_selection

from .load_data import RSFDataLoader

def compile_default_rsfs_data(data_loader):
    VAL_LIQ_FNAME = "rsf_data_liq_11000.dat"
    VAL_ICE_FNAME = "rsf_data_ice_11000.dat"

    df_train, y_train = data_loader.load_rsfs_liq_sol()
    df_val, y_val = data_loader.load_rsfs_liq_sol(liq_fname=VAL_LIQ_FNAME, sol_fname=VAL_ICE_FNAME)

    return X_train, y_train, X_val, y_val

def simple_SVM(X_train, y_train, X_val, y_val=None, C=10, message=None):
    clf = svm.LinearSVC() if not C else svm.LinearSVC(C=C)
    clf.fit(X_train, y_train)
    val_ypred = clf.predict(X_val)
    
    print("Linear SVM with C={}".format(C))
    if message:
        print('\t' + message)
    if not y_val:
        print("\t{}% positive predictions".format(np.sum(val_ypred) / len(val_ypred)))
    else:
        err = np.sum(np.abs(val_ypred - y_val)) / X_val.shape[0]
        print("\terror was %d", err)

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

def plot_SVM_confidence(X, y, sep_val_confidence=False, X_val=None, y_val=None, nbins=100, fname="confidence.png", title="Decision Function"):
    if not sep_val_confidence:
        X_val = X
        y_val = y
    clf = svm.LinearSVC(C=10)
    clf.fit(X,y)
    confidence = clf.decision_function(X_val)
    maxconf = confidence.max()
    minconf = confidence.min()
    
    bins = np.linspace(minconf, maxconf, nbins)
    plt.hist(confidence[y_val==1], bins=bins, alpha=.5, label="liq", edgecolor="k")
    plt.hist(confidence[y_val==0], bins=bins, alpha=.6, label="ice", edgecolor="k")
    plt.title(title)
    plt.legend(loc="best")
    plt.savefig(fname)
    plt.clf()

def main():
    data_loader = RSFDataLoader()
    X_train, y_train, X_val, y_val, hdrs = compile_default_rsfs_data(data_loader)