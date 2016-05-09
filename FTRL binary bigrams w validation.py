# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:05:38 2015

@author: subhajit
"""

#############################################################################################################
#classic tinrtgu's code
#https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory
#modified by rcarson
#remodified by subhajit
#https://www.kaggle.com/jiweiliu
#############################################################################################################


import os, glob
import re, string
from nltk.stem import PorterStemmer
from collections import defaultdict
from datetime import datetime
import pandas as pd
from math import exp, log, sqrt
from random import random
import numpy as np
from sklearn.metrics import roc_auc_score
import pickle

os.chdir('D:\Data Science Competitions\Kaggle\Dato Truly Native\codes')

inputDir = '../input'
fIn = glob.glob( inputDir + '/**/*raw*')
fId = [f.split('\\')[-1] for f in fIn]
fIn = pd.DataFrame(fIn, index=fId)
fIn.columns = ['filePath']

train = pd.read_csv('../input/train_v2.csv', index_col=0)
train = pd.merge(train, fIn, left_index=True, right_index=True)

test = pd.read_csv('../input/sampleSubmission_v2.csv', index_col=0)
test = pd.merge(test, fIn, left_index=True, right_index=True)

np.random.seed(999)
r = np.random.rand(train.shape[0])
valid = train[r>0.99]
train = train[r<=0.99]

# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there


##############################################################################
# parameters #################################################################
##############################################################################

# A, paths
submission = '../output/ftrl1sub.csv'  # path of to be outputted submission file

# B, model
alpha = .005  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 0.     # L1 regularization, larger value means more regularized
L2 = 1.     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 24             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

# D, training/validation
epoch = 1       # learn training data for N passes
holdafter = 9   # data after date N (exclusive) are used as validation
holdout = None  # use every N training instance for holdout validation

regex1 = re.compile('[%s]' % re.escape(string.punctuation))
regex2 = re.compile('\s')
stemmer = PorterStemmer()

##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [random() for k in range(D)]#[0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x, xval):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if self.z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * self.z[i] <= self.L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * self.L1 - self.z[i]) / ((self.beta + sqrt(self.n[i])) / self.alpha + self.L2)

            wTx += w[i]*xval[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, xval, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # update z and n
        for i in self._indices(x):
            # gradient under logloss
            g = (p - y)*xval[i]
            sigma = (sqrt(self.n[i] + g * g) - sqrt(self.n[i])) / self.alpha
            self.z[i] += g - sigma * self.w[i]
            self.n[i] += g * g


def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

def remove_space_punct(text):
    text = regex1.sub(' ',text)
    text = regex2.sub(' ',text)  
    return text

def get_bigrams(wordlist):
    w1 = wordlist[:-1]
    w2 = wordlist[1:]
    n = len(wordlist) - 1
    bigrams = ['%s %s' % (w1[i],w2[i]) for i in range(n)]
    return bigrams

def data(train_or_test, D):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    for idx, row in train_or_test.iterrows():
        # process id
        #print row
    
        with open(row['filePath'], 'r') as page:
           doc = page.read().replace('\n',' ')
        doc = doc.lower()
        doc = remove_space_punct(doc)
        
        # process clicks
        target='sponsored'#'IsClick' 
        y = row[target]

        # extract date

        # turn hour really into hour, it was originally YYMMDDHH

        # build x
        words = re.findall('\w+',doc)
        # words = [stemmer.stem(word) for word in words]
        words = words + get_bigrams(words)
        
        x = []
        xval = defaultdict(float)
        for word in words:
            # one-hot encode everything with hash trick
            index = abs(hash(word)) % D
            x.append(index)
            xval[index] += 1
        
        for key in xval:
            xval[key] = 0 if xval[key]==0 else 1
        
        x = list(set(x))
        xval[0] = 1. #for intercept

        yield idx, x, xval, y


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

# start training
for e in range(epoch):
    loss = 0.
    count = 0
    for file, x, xval, y in data(train, D):  # data is a generator

        p = learner.predict(x, xval)
        loss += logloss(p, y)
        learner.update(x, xval, p, y)
        count+=1
        if count%1000==0:
            #print count,loss/count
            print('%s\tencountered: %d\tcurrent logloss: %f' % (
                datetime.now(), count, loss/count))
        if count>1000000: # comment this out when you run it locally.
            break

#import pickle
pickle.dump(learner,open('../output/ftrl.p','wb'))
#learner = pickle.load(open('../output/ftrl.p','rb'))
        
count=0
loss=0
vprobs = []
print ('validation')
##############################################################################
#################### start validation ########################################
##############################################################################

for  file, x, xval, y in data(valid, D):
    count+=1
    p = learner.predict(x, xval)
    vprobs.append(p)
    loss += logloss(p, y)

    if count%1000==0:
        #print count,loss/count
        print('%s\tencountered: %d\tcurrent logloss: %f' % (
            datetime.now(), count, loss/count))
            
vprobs = np.array(vprobs)
vsponsored = np.array(valid.sponsored)
print('validation score: %f' % roc_auc_score(vsponsored, vprobs))

valid['sponsored'] = vprobs
valid.drop('filePath', axis=1, inplace=True)
valid.reset_index(inplace=True)
valid.columns = ['file','sponsored']
valid.to_csv('../output/validation.csv', index=False)


#import pickle
#pickle.dump(learner,open('../output/ftrl.p','wb'))
#learner = pickle.load(open('../output/ftrl.p','rb'))

count=0
loss=0
print ('write result')
##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################
with open(submission, 'w') as outfile:
    outfile.write('file,sponsored\n')
    for  file, x, xval, y in data(test, D):
        count+=1
        p = learner.predict(x, xval)
        loss += logloss(p, y)

        outfile.write('%s,%s\n' % (file, str(p)))
        if count%1000==0:
            #print count,loss/count
            print('%s\tencountered: %d\tcurrent logloss: %f' % (
                datetime.now(), count, loss/count))
                
ftrlsub = pd.read_csv(submission, index_col=0)
sampleSubmission = pd.read_csv('../input/sampleSubmission_v2.csv', index_col=0)
submission = pd.merge(sampleSubmission, ftrlsub, how='left', left_index=True, right_index=True)
submission.drop('sponsored_x', axis=1, inplace=True)
submission.reset_index(inplace=True)
submission.columns = ['file','sponsored']
submission.fillna(0,inplace=True)
submission.to_csv('../output/submission_ftrl.csv', index=False)