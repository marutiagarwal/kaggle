#!/usr/bin/python
import pandas as pd
import numpy as np
from copy import deepcopy
import time
import dateutil.parser as dateparser
import re
import math
from sklearn.preprocessing import Imputer
from matplotlib.pyplot import plot, show

# http://wavedatalab.github.io/datawithpython/visualize.html
# http://pandas.pydata.org/pandas-docs/stable/missing_data.html
# 
# 
# import time
# import dateutil.parser as dateparser
# datetimestring = '20DEC09:00:00:00'
# timestamp = time.mktime(time.strptime(datetimestring, '%d%b%y:%H:%M:%S'))

# NOTE:
# https://www.kaggle.com/c/springleaf-marketing-response/forums/t/16082/var-0434-var-0449

# Missing-Features: 
# You can either remove the samples with missing features or 
# replace the missing features with their column-wise medians or means.
# 
# 
# http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values
# http://fastml.com/condfting-categorical-data-into-numbers-with-pandas-and-scikit-learn/
# 
# 1. if a feature is missing for most of the rows -> delete that feature
# 2. if a row has most of the features missing -> delete that row 
# 3. Some columns might be just copy of each other -> remove redundant ones
# 4. There are bunch of (NA, -1, "",[]) features to take care of


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

	row = 0
	for x in xrange(0,ncols):
		# print 'x = ',x,', type = ',type(X[row,x]),', X[row,x] = ',X[row,x]
		if type(X[row,x]) is float and math.isnan(float(X[row,x])):
			print 'math.isnan'
		# while math.isnan(float(X[row,x])):
		# 	row += 10
		# 	print 'X[row,x] = ',X[row,x]
		if(type(X[row,x]) is str):
			# print 'x = ',x,', type = ',type(X[row,x])
			# print 'X[row,x] = ',X[row,x]
			datetimestring = X[row,x]
			line = datetimestring[-9:]
			ret = re.match(':\d{2}:\d{2}:\d{2}',line)
			if ret is not None:
				count_timestamp += 1
				idx_timestamp.append(x)
				continue
			count_str += 1
			idx_str.append(x)
		elif(type(X[row,x]) is bool):
			count_bool += 1	
			idx_bool.append(x)		
		elif(type(X[row,x]) is int):
			count_int += 1
			idx_int.append(x)
		elif(type(X[row,x]) is float):
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
	# print 'dict_str = ',dict_str
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

	# condft str feature->int
	_X = deepcopy(X)


def fill_missing_numerics(df, method='mean'):
	for f in df.columns:
		if(df[f].dtype == np.int64 or df[f].dtype == np.float64):
			# print 'f = ',f,', dtype = ',df[f].dtype			
			if (method=='mean'):
				# print 'df[f].mean() = ',df[f].mean()
				df[f].fillna(df[f].mean())
			else:
				# print 'df[f].median() = ',df[f].median()
				df[f].fillna(df[f].median())				
	return df


def delete_all_zero_cols(df):
	# delete all zero cols
	df = df.loc[:, (df != 0).any(axis=0)]
	return df


def delete_all_nan_cols_rows(df):
	# delete all nan cols
	df = df.dropna(axis=0, how='all')

	# delete all nan rows
	df = df.dropna(axis=1, how='all')
	return df


def perprocess_train_features(df, labels):
	# Junk cols - Some feature engineering needed here
	# df = df.ix[:, 520:660].fillna(-1)
	# df = df.ix[:, :].fillna(-1)
	# print 'df.shape = ',df.shape
	# numExamples, numFeatures = df.shape
	# featureNames = df.columns
	# print 'df.dtypes = ',df.dtypes
	
	# print 'VAR_0001 = ',df['VAR_0001'].unique()
	
	# Get descriptive statistics for a specified column
	# print df.VAR_0019.describe()

	# print 'df.VAR_0001.dtype = ',df.VAR_0001.dtype
	# print 'df.VAR_0002.dtype = ',df.VAR_0002.dtype
	# print 'df.VAR_0002.mean() = ',df.VAR_0002.mean()
	# print pd.Series.isnull(df['VAR_1934'])

	print 'df.get_dtype_counts() = \n',df.get_dtype_counts()

	# "delete" the zero-columns
	print 'before deletion: df.shape = ',df.shape
	df = delete_all_zero_cols(df)
	print 'after deletion: df.shape = ',df.shape
	df = delete_all_nan_cols_rows(df)
	print 'after deletion: df.shape = ',df.shape

	# take care of missing numerics
	# df = fill_missing_numerics(df)

	# Change all NaNs to None
	# df = df.where((pd.notnull(df)), None)

	# plot histogram
	# df.VAR_0020.value_counts().plot(kind='bar')
	# show()

	# Check a boolean condition
	# print (df.ix[:,'VAR_1933'] > 9998).any()


	X = df.values.copy()
	# print X
	# impute missing values
	# imp = Imputer(missing_values='NA', strategy='most_frequent', axis=0)
	# imp.fit(X)
	# X_imp = imp.transform(X)

	# type of features
	# feature_type(X)
	# analyze_str_features(X)

    # np.random.shuffle(X)
	X = X.astype(np.float32)

	encoder = LabelEncoder()
	y = encoder.fit_transform(labels).astype(np.int32)
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	return X, y, encoder, scaler

