"""
Author: Louí Byrne
Date: 06/11/2022
Course: Machine Learning
University: KTH
Title: Boosted Decision Tree solution to classification problem

"""

import math
import random
import numpy as np
from numpy import genfromtxt
from sklearn import decomposition, tree
import pandas as pd


# Splits data into training and test set, pcSplit defines the percent of
# the data should be used as training data. The major difference to
# trteSplit is that we select the percent from each class individually.
# This means that we are assured to have enough points for each class.
def trteSplitEven(X,y,pcSplit,seed=None):
    labels = np.unique(y)
    xTr = np.zeros((0,X.shape[1]))
    xTe = np.zeros((0,X.shape[1]))
    yTe = np.zeros((0,),dtype=int)
    yTr = np.zeros((0,),dtype=int)
    trIdx = np.zeros((0,),dtype=int)
    teIdx = np.zeros((0,),dtype=int)
    np.random.seed(seed)
    for label in labels:
        classIdx = np.where(y==label)[0]
        NPerClass = len(classIdx)
        Ntr = int(np.rint(NPerClass*pcSplit))
        idx = np.random.permutation(NPerClass)
        trClIdx = classIdx[idx[:Ntr]]
        teClIdx = classIdx[idx[Ntr:]]
        trIdx = np.hstack((trIdx,trClIdx))
        teIdx = np.hstack((teIdx,teClIdx))
        # Split data
        xTr = np.vstack((xTr,X[trClIdx,:]))
        yTr = np.hstack((yTr,y[trClIdx]))
        xTe = np.vstack((xTe,X[teClIdx,:]))
        yTe = np.hstack((yTe,y[teClIdx]))

    return xTr,yTr,xTe,yTe,trIdx,teIdx


# Read in the data
def fetchDataset(dataset='challenge'):
    if dataset == 'challenge':
        # Read in dataset from csv file
        data = genfromtxt('TrainOnMe-4.csv', delimiter=',', skip_header=1, encoding='UTF-8', invalid_raise=False, dtype=None)
        # Convert to numpy array
        data_df = pd.DataFrame(data)
        data_np = data_df.to_numpy()

        # Extract Independent variables - X
        X = data_np[:, 2:]
        # Clean up dataset
        X[np.where(X == '?')] = 0
        X[np.where(X == 'Slängpolskor')] = 1
        X[np.where(X == 'Hambo')] = 2
        X[np.where(X == 'Schottis')] = 3
        X[np.where(X == 'Polka')] = 4
        X[np.where(X == 'Polskor')] = 5
        X[np.where(X == 'olka')] = 6
        X[np.where(X == 'chottis')] = 7
        X[np.where(X == 'True')] = 1
        X[np.where(X == 'YEP True')] = 1
        X[np.where(X == 'False')] = 0
        X[np.where(X == 'Nope False')] = 0
        X = np.vstack(X[:, :]).astype(float)

        # Extract Dependant Variables - y
        y = data_np[:, 1]
        y[np.where(y == 'Atsuto')] = 0
        y[np.where(y == 'Bob')] = 1
        y[np.where(y == 'Jorg')] = 2
        y = y.astype('int')

        # Read in dataset to be evaluated
        data_evaluate = genfromtxt('EvaluateOnMe-4.csv', delimiter=',', skip_header=1, encoding='UTF-8',
                                   invalid_raise=False, dtype=None)
        data_evaluate_df = pd.DataFrame(data_evaluate)
        data_evaluate_np = data_evaluate_df.to_numpy()
        X_eval = data_evaluate_np[:, 1:]
        X_eval[np.where(X_eval == '?')] = 0
        X_eval[np.where(X_eval == 'Slängpolskor')] = 1
        X_eval[np.where(X_eval == 'Hambo')] = 2
        X_eval[np.where(X_eval == 'Schottis')] = 3
        X_eval[np.where(X_eval == 'Polka')] = 4
        X_eval[np.where(X_eval == 'Polskor')] = 5
        X_eval[np.where(X_eval == 'olka')] = 6
        X_eval[np.where(X_eval == 'chottis')] = 7
        X_eval[np.where(X_eval == 'True')] = 1
        X_eval[np.where(X_eval == 'YEP True')] = 2
        X_eval[np.where(X_eval == 'False')] = 3
        X_eval[np.where(X_eval == 'Nope False')] = 4
        X_eval = np.vstack(X_eval[:, :]).astype(float)

        pcadim = 0

    else:
        print("Please specify a dataset!")
        X = np.zeros(0)
        y = np.zeros(0)
        pcadim = 0

    return X, y, X_eval, pcadim


# The function below, `testClassifier`, will be used to try out the different datasets.
def testClassifier(classifier, dataset='challenge', dim=0, split=0.7, ntrials=100):

    X,y,X_eval,pcadim = fetchDataset(dataset)

    means = np.zeros(ntrials,);

    for trial in range(ntrials):

        xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split,trial)

        # Do PCA replace default value if user provides it
        if dim > 0:
            pcadim = dim

        if pcadim > 0:
            pca = decomposition.PCA(n_components=pcadim)
            pca.fit(xTr)
            xTr = pca.transform(xTr)
            xTe = pca.transform(xTe)

        # Train
        trained_classifier = classifier.trainClassifier(xTr, yTr)
        # Predict
        yPr = trained_classifier.classify(xTe)

        # Compute classification error
        if trial % 10 == 0:
            print("Trial:",trial,"Accuracy","%.3g" % (100*np.mean((yPr==yTe).astype(float))) )

        means[trial] = 100*np.mean((yPr==yTe).astype(float))

    print("Final mean classification accuracy ", "%.3g" % (np.mean(means)), "with standard deviation", "%.3g" % (np.std(means)))

    mean_of_means = np.mean(means)
    return mean_of_means


# Classify the X_test dataset using ALL of the training data
def classify_new_dataset(classifier, dataset='challenge'):

    X, y, X_eval, pcadim = fetchDataset(dataset)

    # Train
    trained_classifier = classifier.trainClassifier(X, y)
    # Predict
    yPr = trained_classifier.classify(X_eval)

    return yPr


class DecisionTreeClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = DecisionTreeClassifier()
        rtn.classifier = tree.DecisionTreeClassifier(max_depth=round(Xtr.shape[1]/2+1))
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)


# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts, 1)) / Npts
    else:
        assert (W.shape[0] == Npts)

    classes = np.unique(labels)
    Nclasses = np.size(classes)
    prior = np.zeros((Nclasses, 1))
    class_count = np.zeros(Nclasses)

    # Compute the values of prior for each class!
    for i in range(Npts):
        k = labels[i]
        class_count[k] += W[i]
    for k in range(Nclasses):
        prior[k] = class_count[k] / np.sum(W)

    return prior


# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W=None):
    assert (X.shape[0] == labels.shape[0])
    Npts, Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    mu = np.zeros((Nclasses, Ndims))
    sigma = np.zeros((Nclasses, Ndims, Ndims))

    class_count = np.zeros(Nclasses)

    if W is None:
        W = np.ones((Npts, 1)) / float(Npts)

    # Compute mu and sigma
    for i in range(Npts):
        k = labels[i]
        mu[k] += X[i] * W[i]
        class_count[k] += W[i]
    for k in range(Nclasses):
        mu[k] /= class_count[k]
    for i in range(Npts):
        k = labels[i]
        for m in range(Ndims):
            sigma[k][m][m] += W[i] * np.power(X[i][m] - mu[k][m], 2)
    for k in range(Nclasses):
        for m in range(Ndims):
            sigma[k][m][m] /= class_count[k]

    return mu, sigma


# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):
    Npts = X.shape[0]
    Nclasses, Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    # Compute the log posterior logProb:
    # ==========================
    for k in range(Nclasses):
        for i in range(Npts):
            ln_det_sigma = -0.5 * np.log(np.linalg.det(sigma[k]))
            ln_prior = np.log(prior[k])
            diff = X - mu[k]
            logProb[k][i] = ln_det_sigma - 0.5 * np.inner(diff[i] / np.diag(sigma[k]), diff[i]) + ln_prior
    # ==========================

    h = np.argmax(logProb, axis=0)
    return h


class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    Npts, Ndims = np.shape(X)
    classifiers = []  # append new classifiers to this list
    alphas = []  # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts, 1)) / float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # Calculate error and Alpha
        deltas = delta(vote, labels)
        error = 0
        for i in range(Npts):
            error += wCur[i] * (1 - deltas[i])
        alpha = 0.5 * (np.log(1 - error) - np.log(error))  # Compute new alpha
        alphas.append(alpha)  # you will need to append the new alpha

        # Update weights
        wOld = wCur
        for i in range(len(wCur)):
            if deltas[i] == 1:
                wCur[i] = wOld[i] * math.e ** (-alpha)
            else:
                wCur[i] = wOld[i] * math.e ** (alpha)
        wCur = wCur / np.sum(wCur)

    return classifiers, alphas


def delta(prediction, labels):
    assert (prediction.shape == labels.shape)
    out = np.zeros_like(labels)
    for i in range(len(labels)):
        if prediction[i] == labels[i]:
            out[i] = 1
    return out


# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts, Nclasses))

        # here we can do it by filling in the votes vector with weighted votes
        for idx, classifier in enumerate(classifiers):
            classifiction_res = classifier.classify(X)
            for i in range(Npts):
                votes[i][classifiction_res[i]] += alphas[idx]

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes, axis=1)


class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# Naive Bayes - Test & Plot
#testClassifier(BayesClassifier(), split=0.7)

# Boosted Naive Bayes - Test & Plot
#testClassifier(BoostClassifier(BayesClassifier(), T=30), split=0.7)

# Boosted Decision Tree - Test & Plot
#testClassifier(DecisionTreeClassifier(), split=0.7)
#mean = testClassifier(BoostClassifier(DecisionTreeClassifier(), T=100), split=0.7)

# BEST PERFORMING CLASSIFIER = Boosted Decision Tree, mean accuracy = 73.9% when using cross-validation
# Predict the classifications of the evaluation dataset
ypr = classify_new_dataset(BoostClassifier(DecisionTreeClassifier(), T=100))
# Convert the class placeholders back to strings
y_out = np.zeros(len(ypr), dtype=object)
for i in range(len(ypr)):
    if ypr[i] == 0:
        y_out[i] = 'Atsuto'
    elif ypr[i] == 1:
        y_out[i] = 'Bob'
    elif ypr[i] == 2:
        y_out[i] = 'Jorg'

y_out = np.vstack(y_out[:]).astype(str)

# Save the predicted y to a .txt file
np.savetxt("labels.txt", y_out, delimiter=",", fmt='%s')
