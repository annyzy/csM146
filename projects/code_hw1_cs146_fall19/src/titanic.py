"""
Description : Titanic
"""

## IMPORTANT: Use only the provided packages!

## SOME SYNTAX HERE.
## I will use the "@" symbols to refer to some variables and functions.
## For example, for the 3 lines of code below
## x = 2
## y = x * 2
## f(y)
## I will use @x and @y to refer to variable x and y, and @f to refer to function f

import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics


######################################################################
# classes
######################################################################

class Classifier(object) :

    ## THIS IS SOME GENERIC CLASS, YOU DON'T NEED TO DO ANYTHING HERE.

    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) : ## INHERITS FROM THE @CLASSIFIER

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        # n,d = X.shape ## get number of sample and dimension
        y = [self.prediction_] * X.shape[0]
        return y


class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- an array specifying probability to survive vs. not
        """
        self.probabilities_ = None ## should have length 2 once you call @fit

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        # in simpler wordings, find the probability of survival vs. not

        ## count the probability if y = 1;
        count_y = Counter(y)
        self.probabilities_ = float(count_y[1.0])/(float(count_y[0.0])+float(count_y[1.0]))

        ### ========== TODO : END ========== ###

        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (check the arguments of np.random.choice) to randomly pick a value based on the given probability array @self.probabilities_
        y = np.random.choice(2, X.shape[0],p=[1-self.probabilities_, self.probabilities_])

        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2, train_size=None):
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use @train_test_split to split the data into train/test set
    # xtrain, xtest, ytrain, ytest = train_test_split (X,y, test_size = test_size, random_state = i)
    # now you can call the @clf.fit (xtrain, ytrain) and then do prediction

    train_error = 0 ## average error over all the @ntrials
    test_error = 0
    train_scores = [];
    test_scores = []; ## tracking the error for each of the @ntrials, these array should have length 100 once you're done.
    for i in range(0,ntrials):
        xtrain, xtest, ytrain, ytest = train_test_split (X,y, test_size = test_size, train_size=train_size, random_state = i)
        clf.fit (xtrain, ytrain)
        y_pred = clf.predict(xtrain)
        # train_scores.append(train_error)
        # test_scores.append(test_error)
        # train_error += 1-metrics.accuracy_score(ytrain, y_pred, normalize = True)
        # y_pred = clf.predict(xtest)
        # test_error += 1-metrics.accuracy_score(ytest, y_pred, normalize = True)
        train_error = 1-metrics.accuracy_score(ytrain, y_pred, normalize = True)
        train_scores.append(train_error)
        y_pred = clf.predict(xtest)
        test_error = 1-metrics.accuracy_score(ytest, y_pred, normalize = True)
        test_scores.append(test_error)

    Sum1 = sum(train_scores)
    train_error = Sum1/ntrials

    Sum2 = sum(test_scores)
    test_error = Sum2/ntrials

    ### ========== TODO : END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # part a: plot histograms of each feature
    # print('Plotting...')
    # for i in range(d) :
    #     plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)
    #
    #
    # #========================================
    # # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)



    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    rc_clf = RandomClassifier() # create RandomClassifier classifier, which includes all model parameters
    rc_clf.fit(X, y)                  # fit training data using the classifier
    y_pred = rc_clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    print('Classifying using Decision Tree...')
    # call the function @DecisionTreeClassifier
    dtc_clf = DecisionTreeClassifier(criterion="entropy")
    dtc_clf.fit(X, y)
    y_pred = dtc_clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)

    ### ========== TODO : END ========== ###



    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf")
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors


    print('Classifying using k-Nearest Neighbors...')
    # call the function @KNeighborsClassifier
    nn3_clf = KNeighborsClassifier(n_neighbors=3)
    nn3_clf.fit(X, y)
    y_pred = nn3_clf.predict(X)
    train_error = 1 - nn3_clf.fit (X,y).score(X,y)
    print('\t-- training error for 3-NN : %.3f' % train_error)

    nn5_clf = KNeighborsClassifier(n_neighbors=5)
    nn5_clf.fit(X, y)
    y_pred = nn5_clf.predict(X)
    train_error = 1 - nn5_clf.fit (X,y).score(X,y)
    print('\t-- training error for 5-NN : %.3f' % train_error)

    nn7_clf = KNeighborsClassifier(n_neighbors=7)
    nn7_clf.fit(X, y)
    y_pred = nn7_clf.predict(X)
    train_error = 1 - nn7_clf.fit (X,y).score(X,y)
    print('\t-- training error for 7-NN : %.3f' % train_error)
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    # call your function @error
    part4_clf = MajorityVoteClassifier()
    train_error, test_error = error(part4_clf, X, y)
    print('\t-- MajorityVoteClassifier for training error: %.3f \t testing error: %.3f' % (train_error, test_error))

    part4_clf = RandomClassifier()
    train_error, test_error = error(part4_clf, X, y)
    print('\t-- RandomClassifier for training error:       %.3f \t testing error: %.3f' % (train_error, test_error))

    part4_clf = DecisionTreeClassifier()
    train_error, test_error = error(part4_clf, X, y)
    print('\t-- DecisionTreeClassifier for training error: %.3f \t testing error: %.3f' % (train_error, test_error))

    part4_clf = KNeighborsClassifier(n_neighbors=5)
    train_error, test_error = error(part4_clf, X, y)
    print('\t-- KNeighborsClassifier for training error:   %.3f \t testing error: %.3f' % (train_error, test_error))
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    # the algorithm of below is suggested by TA and the way to plot the graph is accomdated online
    print('Finding the best k for KNeighbors classifier...')
    x_plot=[]
    y_plot=[]
    # hint: use the function @cross_val_score
    k = list(range(1,50,2))
    cv_score = [] ## track accuracy for each value of $k, should have length 25 once you're done
    for i in k:
        partf_clf = KNeighborsClassifier(n_neighbors=i)
        scores = cross_val_score(partf_clf, X, y, cv=10, scoring='accuracy')
        print('\t-- 10-fold cross validation error for %d-NN: %.3f' % (i, 1-np.mean(scores)))
        x_plot.append(i)
        y_plot.append(1-scores.mean())

    plt.plot(x_plot, y_plot, '--')
    plt.axis('auto')
    plt.xlabel('K')
    plt.ylabel('Average Error of 10-Fold Cross Validation')
    plt.title('K-NN Classifier of 10-Fold Cross Validation')
    plt.savefig("4.2-figure_f.pdf")
    plt.clf()
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    x_plot=[]
    trainingError_plot=[]
    testingError_plot=[]
    k = list(range(1,21,1))

    for i in k:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=i)
        train_error, test_error= error(clf, X, y)
        print('\t-- Cross Validation of Decision Tree Classifier on Depth Limits %d for training error: %.3f \t testing error: %.3f' % (i, train_error, test_error))
        x_plot.append(i)
        trainingError_plot.append(train_error)
        testingError_plot.append(test_error)

    blue, =plt.plot(x_plot, trainingError_plot, '--', label='Average Training Error')
    orange, =plt.plot(x_plot, testingError_plot, '--', label='Average Test Error')
    plt.axis('auto')
    plt.xlabel('Depth limits')
    plt.ylabel('Average Error')
    plt.legend(loc='lower left')
    plt.title('Cross Validation of Decision Tree Classifier on Depth Limits')
    plt.savefig("4.2-figure_g.pdf")
    plt.clf()
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    x_plot=[]
    dTTrainErrorPlot=[]
    dTTestErrorPlot=[]
    clfDT = DecisionTreeClassifier(criterion='entropy', max_depth=6)
    kNNTrainErrorPlot=[]
    kNNTestErrorPlot=[]
    clfKNN = KNeighborsClassifier(n_neighbors=7)
    k = list(range(1,11,1))

    for i in k:
        train_error, test_error= error(clfDT, X, y, test_size=0.1, train_size=i*0.09)
        print('\t-- Depth Limits(6) Decision Tree %d%% for average training error:    %.3f \t testing error: %.3f' % (i*10, train_error, test_error))
        x_plot.append(i*10)
        dTTrainErrorPlot.append(train_error)
        dTTestErrorPlot.append(test_error)
        train_error, test_error= error(clfKNN, X, y, test_size=0.1, train_size=i*0.09)
        print('\t-- KNN(7) of %d%% for                     average training error:    %.3f \t testing error: %.3f' % (i*10, train_error, test_error))
        kNNTrainErrorPlot.append(train_error)
        kNNTestErrorPlot.append(test_error)

    line1, =plt.plot(x_plot, dTTrainErrorPlot, '--', label='Decision Tree Training Error')
    line2, =plt.plot(x_plot, dTTestErrorPlot, '--', label='Decision Tree Test Error')
    line3, =plt.plot(x_plot, kNNTrainErrorPlot, '--', label='KNN Training Error')
    line4, =plt.plot(x_plot, kNNTestErrorPlot, '--', label='KNN Test Error')
    plt.axis('auto')
    plt.xlabel('Training Data Set Size')
    plt.ylabel('Average Error')
    plt.legend(loc='lower left')
    plt.savefig("4.2-figure_h.pdf")
    plt.clf()
    ### ========== TODO : END ========== ###


    print('Done')


if __name__ == "__main__":
    main()
