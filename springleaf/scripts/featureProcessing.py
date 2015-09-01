#!/usr/bin/python
import pandas as pd
import numpy as np


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

