#!/usr/bin/python
import pandas as pd
import numpy as np
from copy import deepcopy
import time
import dateutil.parser as dateparser
import re

# import time
# import dateutil.parser as dateparser
# datetimestring = '20DEC09:00:00:00'
# timestamp = time.mktime(time.strptime(datetimestring, '%d%b%y:%H:%M:%S'))

def feature_type(X):
	nrows, ncols = X.shape
	count_str = 0
	count_int = 0
	count_bool = 0
	count_float = 0
	count_timestamp = 0
	idx_str = []
	idx_bool = []
	idx_int = []
	idx_float = []
	idx_timestamp = []
	for x in xrange(0,ncols):
		# print 'x = ',x,', type = ',type(X[1,x])	
		if(type(X[1,x]) is str):
			if(len(X[1,x])>8):
				print 'x = ',x,', type = ',type(X[1,x])
				print 'X[1,x] = ',X[1,x]
				datetimestring = X[1,x]
				line = datetimestring[-8:]
				ret = re.match('\d{2}:\d{2}:\d{2}',line)
				if ret is not None:
					count_timestamp += 1
					idx_timestamp.append(x)
					continue
			count_str += 1
			idx_str.append(x)
		elif(type(X[1,x]) is bool):
			count_bool += 1	
			idx_bool.append(x)		
		elif(type(X[1,x]) is int):
			count_int += 1
			idx_int.append(x)
		elif(type(X[1,x]) is float):
			count_float += 1
			idx_float.append(x)
	print 'count_str = ',count_str,', count_int = ',count_int,', count_bool = ',count_bool,', count_float = ',count_float,', count_timestamp = ',count_timestamp
	print 'idx_timestamp = ',idx_timestamp


def analyze_str_features(X):
	nrows, ncols = X.shape
	dict_str = dict()
	for x in xrange(0,ncols):
		for y in xrange(0,nrows):
			if(type(X[y,x]) is str):
				if x not in dict_str:
					dict_str[x] = []
				if X[y,x] not in dict_str[x]:
					dict_str[x].append(X[y,x])
	print 'dict_str = ',dict_str
	# encode_str_features(X, dict_str)


def encode_str_features(X, dict_str):
	# prepare str feature->int mapping
	dict_str2int = dict()
	for idx_feat in dict_str:
		if idx_feat not in dict_str2int:
			dict_str2int[idx_feat] = dict()		

		features = dict_str[idx_feat]
		for x,f in enumerate(features):
			if f not in dict_str2int[idx_feat]:
				dict_str2int[idx_feat][f] = dict()
			dict_str2int[idx_feat][f] = x

	print 'dict_str2int = ',dict_str2int

	# time-stamp features' index
	timestamp = [155, ]

	# convert str feature->int
	_X = deepcopy(X)


def perprocess_train_features(df):
	# Junk cols - Some feature engineering needed here
	# df = df.ix[:, 520:660].fillna(-1)
	# df = df.ix[:, :].fillna(-1)

	X = df.values.copy()

	# type of features
	feature_type(X)
	analyze_str_features(X)

    # np.random.shuffle(X)
	X = X.astype(np.float32)

	encoder = LabelEncoder()
	y = encoder.fit_transform(labels).astype(np.int32)
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	return X, y, encoder, scaler

