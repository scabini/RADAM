# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:39:21 2022

LDA in torch

from https://github.com/liyu10000/DeepLDA/blob/master/lda.py
from the paper "Deep Linear Discriminant Analysis"
"""

import torch
import torch.nn as nn
from functools import partial


def lda(X, y, n_classes, lamb):
    # flatten X
    X = X.view(X.shape[0], -1)
    N, D = X.shape

    # count unique labels in y
    labels, counts = torch.unique(y, return_counts=True)
    assert len(labels) == n_classes  # require X,y cover all classes

    # compute mean-centered observations and covariance matrix
    X_bar = X - torch.mean(X, 0)
    Xc_mean = torch.zeros((n_classes, D), dtype=X.dtype, device=X.device, requires_grad=False)
    St = X_bar.t().matmul(X_bar) / (N - 1)  # total scatter matrix
    Sw = torch.zeros((D, D), dtype=X.dtype, device=X.device, requires_grad=True)  # within-class scatter matrix
    for c, Nc in zip(labels, counts):
        Xc = X[y == c]
        Xc_mean[int(c), :] = torch.mean(Xc, 0)
        Xc_bar = Xc - Xc_mean[int(c), :]
        Sw = Sw + Xc_bar.t().matmul(Xc_bar) / (Nc - 1)
    Sw /= n_classes
    Sb = St - Sw  # between scatter matrix

    # cope for numerical instability
    Sw += torch.eye(D, dtype=X.dtype, device=X.device, requires_grad=False) * lamb

    # compute eigen decomposition
    temp = Sw.pinverse().matmul(Sb)
    # evals, evecs = torch.symeig(temp, eigenvectors=True) # only works for symmetric matrix
    evals, evecs = torch.eig(temp, eigenvectors=True) # shipped from nightly-built version (1.8.0.dev20201015)
    print(evals.shape, evecs.shape)

    # remove complex eigen values and sort
    noncomplex_idx = evals[:, 1] == 0
    evals = evals[:, 0][noncomplex_idx] # take real part of eigen values
    evecs = evecs[:, noncomplex_idx]
    evals, inc_idx = torch.sort(evals) # sort by eigen values, in ascending order
    evecs = evecs[:, inc_idx]
    print(evals.shape, evecs.shape)

    # flag to indicate if to skip backpropagation
    hasComplexEVal = evecs.shape[1] < evecs.shape[0]

    return hasComplexEVal, Xc_mean, evals, evecs


def lda_loss(evals, n_classes, n_eig=None, margin=None):
    n_components = n_classes - 1
    evals = evals[-n_components:]
    # evecs = evecs[:, -n_components:]
    print('evals', evals.shape, evals)
    # print('evecs', evecs.shape)

    # calculate loss
    if margin is not None:
        threshold = torch.min(evals) + margin
        n_eig = torch.sum(evals < threshold)
    loss = -torch.mean(evals[:n_eig]) # small eigen values are on left
    return loss


class LDA(nn.Module):
    def __init__(self, n_classes, lamb):
        super(LDA, self).__init__()
        self.n_classes = n_classes
        self.n_components = n_classes - 1
        self.lamb = lamb
        self.lda_layer = partial(lda, n_classes=n_classes, lamb=lamb)

    def forward(self, X, y):
        # perform LDA
        hasComplexEVal, Xc_mean, evals, evecs = self.lda_layer(X, y)  # CxD, D, DxD

        # compute LDA statistics
        self.scalings_ = evecs  # projection matrix, DxD
        self.coef_ = Xc_mean.matmul(evecs).matmul(evecs.t())  # CxD
        self.intercept_ = -0.5 * torch.diagonal(Xc_mean.matmul(self.coef_.t())) # C

        # return self.transform(X)
        return hasComplexEVal, evals

    def transform(self, X):
        """ transform data """
        X_new = X.matmul(self.scalings_)
        return X_new[:, :self.n_components]

    def predict(self, X):
        logit = X.matmul(self.coef_.t()) + self.intercept_
        return torch.argmax(logit, dim=1)

    def predict_proba(self, X):
        logit = X.matmul(self.coef_.t()) + self.intercept_
        proba = nn.functional.softmax(logit, dim=1)
        return proba

    def predict_log_proba(self, X):
        logit = X.matmul(self.coef_.t()) + self.intercept_
        log_proba = nn.functional.log_softmax(logit, dim=1)
        return log_proba
    
# if __name__ == '__main__':
#     import numpy as np
#     np.set_printoptions(precision=4, suppress=True)
#     from sklearn.datasets import load_iris
#     from sklearn.metrics import accuracy_score

#     features, labels = load_iris(return_X_y=True)
#     print(features.shape, labels.shape)

#     n_classes = 3
#     n_components = n_classes - 1
#     N, D = features.shape  # 150, 4
#     lamb = 0.001
#     n_eig = 2
#     margin = 0.01

#     device = torch.device('cpu:0')
#     features = np.zeros((150, 2048))
#     X = torch.from_numpy(features).to(device)
#     y = torch.from_numpy(labels).to(device)

#     lda_classifier = LDA(n_classes, lamb)
#     _, evals = lda_classifier(X, y)

#     # calculate lda loss
#     loss = lda_loss(evals, n_classes, n_eig, margin)
#     loss.backward()
#     print('finished backward')

#     # use LDA as classifier
#     y_pred = lda_classifier.predict(X)
#     print('accuracy on training data', accuracy_score(y, y_pred))