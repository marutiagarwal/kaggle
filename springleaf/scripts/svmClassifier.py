from sklearn import svm
from sklearn.externals import joblib


def trySVM(X_train, y_train, X_val, y_val):
	print '-------------- training SVM --------------'
	clf = svm.SVC(kernel='linear', probability=True)
	clf.fit(X_train, y_train)
	joblib.dump(clf, 'svm.pkl', compress=9)
	print "training complete..."

	print '-------------- testing SVM --------------'
	resultLabels = clf.predict_proba(X_val)
 	return clf, resultLabels

