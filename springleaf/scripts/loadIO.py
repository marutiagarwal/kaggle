#!/usr/bin/python
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import random
import lasagne

from sklearn import svm
from sklearn.externals import joblib


def load_train_data(path):
	print("Loading Train Data...")
	df = pd.read_csv(path, nrows=1000)

	labels = df.target

	df = df.drop('target',1)
	df = df.drop('ID',1)	
	return df,labels


def perprocess_train_features(df):
	# Junk cols - Some feature engineering needed here
	df = df.ix[:, 520:660].fillna(-1)
	# df = df.fillna(-1)

	X = df.values.copy()    
    # np.random.shuffle(X)
	X = X.astype(np.float32)

	encoder = LabelEncoder()
	y = encoder.fit_transform(labels).astype(np.int32)
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	return X, y, encoder, scaler


def load_test_data(path, scaler):
	print("Loading Test Data...")
	df = pd.read_csv(path, nrows=1000)
	ids = df.ID.astype(str)

	df = df.drop('ID',1)
    
    # Junk cols - Some feature engineering needed here
	df = df.ix[:, 520:660].fillna(-1)
	# df = df.fillna(-1)

	X = df.values.copy()

	X, = X.astype(np.float32),
	X = scaler.transform(X)
	return X, ids


def evaluate(y_val, resultLabels):
	tp = 0
	fp = 0

	for idx,label in enumerate(resultLabels):
	# #      print 'ground truth label = ',test_y[idx]
	   label = label.tolist()
	   print 'max prob = ',max(label)
	   maxIndex = label.index(max(label))        
	   if(maxIndex==y_val[idx] and max(label)>=0.5):
	       tp += 1
	   else:
	       fp += 1
	         
	print 'fp = ',fp
	print 'tp = ',tp
	print 'Precision = ',tp/float(tp+fp)


def trySVM(X_train, y_train, X_val, y_val):
	print '-------------- training SVM --------------'
	clf = svm.SVC(kernel='linear', probability=True)
	clf.fit(X_train, y_train)
	joblib.dump(clf, 'svm.pkl', compress=9)
	print "training complete..."

	print '-------------- testing SVM --------------'
	resultLabels = clf.predict_proba(X_val)
	evaluate(y_val, resultLabels)
 	return clf


if __name__ == '__main__':
	# Divide the training data into (train + val)
	train_data_file = "../data/train.csv"
	num_lines = sum(1 for l in open(train_data_file)) -1 # remove header
	num_lines = 1000
	print 'num_lines = ',num_lines
	trainSize = int(num_lines * 0.75)
	valSize = num_lines - trainSize

	# Training data
	df, labels = load_train_data(train_data_file)
	X, y, encoder, scaler = perprocess_train_features(df)

	print('Number of classes:', len(encoder.classes_))
	num_classes = len(encoder.classes_)
	num_features = X.shape[1]
	# 1934 features originally !

	# Validation data split
	X_train = X[0:trainSize]
	X_val = X[trainSize:]
	y_train = y[0:trainSize]
	y_val = y[trainSize:]

	# randomize training data
	indices = range(0,trainSize)
	random.shuffle(indices)
	X_train = np.array([X_train[i] for i in indices])
	y_train = np.array([y_train[i] for i in indices])
	print("Training --> n_samples: %d, n_features: %d" % X_train.shape)
	print("Validation --> n_samples: %d, n_features: %d" % X_val.shape)

	# SVM classifier
	classifier = trySVM(X_train, y_train, X_val, y_val)

	# Testing data
	X_test, ids = load_test_data("../data/test.csv", scaler)

	# make predictions
    submission = pd.DataFrame(preds, index=ids, columns=['target'])
    submission.to_csv('BTB_Lasagne.csv')
    
    