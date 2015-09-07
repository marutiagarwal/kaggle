#!/usr/bin/python
import pandas as pd
import numpy as np
from copy import deepcopy
import time
import dateutil.parser as dateparser
import re
import math
from sklearn.preprocessing import Imputer
from matplotlib.pyplot import plot, show, xlabel, ylabel

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


def delete_timestamp_cols(df):
	timestamps = ['VAR_0073', 'VAR_0075', 'VAR_0156', 'VAR_0157', 'VAR_0158', 'VAR_0159',
				  'VAR_0166', 'VAR_0167', 'VAR_0168', 'VAR_0169', 'VAR_0176', 'VAR_0177',
				  'VAR_0178', 'VAR_0179', 'VAR_0204', 'VAR_0217']
	
	# print 'df.shape = ',df.shape
	df_cat = df.drop(timestamps, axis=1)
	# print 'df_cat.shape = ',df_cat.shape
	return df_cat

def fill_missing_numeric_features(df, method='mean'):
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


def delete_all_nan_cols_rows(df, frac):
	num_th = int(df.shape[0]*frac) # require this many non-NA value
	# delete all nan cols
	feat1 = df.columns
	df1 = df.dropna(axis=1, how='all', thresh=num_th)
	feat2 = df1.columns
	print "removed nan features = ",[item for item in feat1 if item not in feat2]

	# delete all nan rows
	df2 = df1.dropna(axis=0, how='all')
	return df2


def delete_cols_with_same_features(df):
	feat1 = df.columns
	df_cat = df.loc[:, df.apply(pd.Series.nunique, axis=0) != 1]
	feat2 = df_cat.columns
	print "removed all-same features = ",[item for item in feat1 if item not in feat2]
	return df_cat


def delete_rows_with_same_features(df):
	df_cat = df.loc[:, df.apply(pd.Series.nunique, axis=1) != 1]
	return df_cat


def process_timestamps(df):
	for f in df.columns:
		if(df[f].dtype == np.object):
			# print 'f = ',f,', dtype = ',df[f].dtype
			df[f].value_counts().plot(kind='bar')
			ylabel(f)
			show()
			# datetimestring = str(df[f][0])
			# print 'datetimestring = ',datetimestring
			# line = datetimestring[-9:]
			# ret = re.match(':\d{2}:\d{2}:\d{2}',line)
			# if ret is not None:			
			# 	print 'line = ',line


def delete_cols_with_high_nan(df, _frac):
	numExamples, numFeatures = df.shape
	drop_cols = []
	for f in df.columns:
		frac = df[f].isnull().sum()/float(numExamples)
		# print 'f = ',f,', nan count = ',df[f].isnull().sum(), ', frac = ',frac
		if frac>=_frac:
			drop_cols.append(f)

	# drop columns with very high nan count
	df_cat = df.drop(drop_cols, axis=1)
	return df_cat 


def delete_str_cols_with_high_negatives(df, _frac):
	drop_cols = []
	for f in df.columns:
		if df[f].dtype == np.object:
			if '-1' in df[f].unique():
				frac = df[f].value_counts('-1')['-1']
				print 'f = ',f,', negatives str count = ',frac
				if frac>=_frac:
					drop_cols.append(f)
		elif df[f].dtype == np.int64 or df[f].dtype == np.float64:
			unique_list = []
			for x in df[f].unique():
				if not math.isnan(x):
					unique_list.append(int(x))

			# if features unique_list = [-1,0], delete this feature
			if unique_list.sort()==[-1,0]:
				drop_cols.append(f)
				continue

			if -1 in unique_list:
				# print 'f = ',f, ', unique_list = ',unique_list
				frac = df[f].value_counts('-1')[-1]
				print 'f = ',f,', negatives int count = ',frac
				if frac>=_frac:
					drop_cols.append(f)

	print 'removed high -ve features  = ',drop_cols
	# drop columns with very high negatives count
	df_cat = df.drop(drop_cols, axis=1)
	return df_cat 


def delete_str_cols_with_high_empty(df, _frac):
	drop_cols = []
	for f in df.columns:
		if df[f].dtype == np.object:
			if '' in df[f].unique() or '[]' in df[f].unique():
				frac = df[f].value_counts('-1')['-1']
				print 'f = ',f,', empty count = ',frac
				if frac>=_frac:
					drop_cols.append(f)

	# drop columns with very high negatives count
	df_cat = df.drop(drop_cols, axis=1)
	return df_cat 


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

	# print pd.Series.isnull(df['VAR_1934'])

	print 'df.VAR_0309.dtype = ',df.VAR_0309.dtype
	print 'df.get_dtype_counts() = \n',df.get_dtype_counts()

	# "delete" the zero-columns
	print 'df.shape = ',df.shape
	df = delete_timestamp_cols(df)
	print 'delete_timestamp_cols: df.shape = ',df.shape

	df = delete_all_zero_cols(df)
	print 'delete_all_zero_cols: df.shape = ',df.shape

	df = delete_all_nan_cols_rows(df, 0.9)
	print 'delete_all_nan_cols_rows: df.shape = ',df.shape

	df = delete_cols_with_same_features(df)
	print 'delete_cols_with_same_features: df.shape = ',df.shape

	df = delete_cols_with_high_nan(df, 0.10)
	print 'delete_cols_with_high_nan: df.shape = ',df.shape	

	df = delete_str_cols_with_high_negatives(df, 0.10)
	print 'delete_str_cols_with_high_negatives: df.shape = ',df.shape	

	df = delete_str_cols_with_high_empty(df, 0.10)
	print 'delete_str_cols_with_high_empty: df.shape = ',df.shape	

	print '\n\nremaining cols = \n'
	for f in df.columns:
		print f

	# process_timestamps(df)
	print 'df.get_dtype_counts() = \n',df.get_dtype_counts()

	# take care of missing numerics
	# df = fill_missing_numeric_features(df)

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

