import pandas as pd 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

def import_file():
	columns = ['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi','dpf', 'age', 'diabetes']
	file_name = 'pimaindian.csv'
	df = pd.read_csv(file_name,names=columns)
	return df


def get_train_test_data(df):
	y = df['diabetes'].values
	X = df.drop('diabetes', axis=1).values

	# Create training and test set
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=42)
	return X,y,X_train, X_test, y_train, y_test


def classifier(clf, X,y,X_train, X_test, y_train, y_test):

	# Fit the classifier to the training data
	clf.fit(X_train, y_train)

	# Predict the labels of the test data: y_pred
	y_pred = clf.predict(X_test)

	# Generate the confusion matrix and classification report
	print(confusion_matrix(y_test, y_pred))
	print(classification_report(y_test, y_pred))

	# Compute predicted probabilities: y_pred_prob
	y_pred_prob = clf.predict_proba(X_test)[:,1]

	# Generate ROC curve values: fpr, tpr, thresholds
	fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

	# plot roc curve

	plot_roc_curve(fpr, tpr)

	# Compute and print AUC score
	print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

	# Compute cross-validated AUC scores: cv_auc
	cv_auc = cross_val_score(clf,X,y,cv=5, scoring ='roc_auc')

	# Print list of AUC scores
	print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


def plot_roc_curve(fpr, tpr):
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.show()

def hyperparametertuning_gridSearchCV(clf,param_grid,X,y):

	# Instantiate the GridSearchCV object: logreg_cv
	clf_cv = GridSearchCV(clf, param_grid, cv=5)

	# Fit it to the data

	clf_cv.fit(X, y)


	# Print the tuned parameters and score
	print("Tuned Parameters: {}".format(clf_cv.best_params_)) 
	print("Best score is {}".format(clf_cv.best_score_))

def hyperparametertuning_RandomizedSearchCV(clf, param_grid, X,y):

	# Fit it to the data

	clf_cv= RandomizedSearchCV(clf, param_grid, cv=5)
	clf_cv.fit(X,y)

	# Print the tuned parameters and score
	print("Tuned Parameters: {}".format(clf_cv.best_params_))
	print("Best score is {}".format(clf_cv.best_score_))






if __name__ == '__main__':
	df = import_file()

	# split test train data
	X,y,X_train, X_test, y_train, y_test = get_train_test_data(df)

	# instantiate KNN
	knn = KNeighborsClassifier(n_neighbors= 6)

	# instantiate logisticregression
	logreg = LogisticRegression()

	# classify using knn
	print "knn"
	classifier(knn, X,y,X_train, X_test, y_train, y_test)

	print "LogisticRegression"
	# classify using logistic regression
	classifier(logreg, X,y,X_train, X_test, y_train, y_test)

	# Hyperparameter tuning GridSearchCV
	# knn params for GridSearchCV
	knn = KNeighborsClassifier()
	knn_params = {'n_neighbors': np.arange(1,50)}
	hyperparametertuning_gridSearchCV(knn, knn_params, X,y)

	# logistic regression params for GridSearchCV

	logreg = LogisticRegression()
	c_space = np.logspace(-5, 8, 15)
	logreg_params = {'C': c_space}
	hyperparametertuning_gridSearchCV(logreg,logreg_params, X, y)

	# Hyperparameter turning Randomized
	# RandomizedSearchCV

	# params for decision tree

	print "decision tree"

	# Setup the parameters and distributions to sample from: param_dist
	param_dist = {"max_depth": [3, None],
	              "max_features": randint(1, 9),
	              "min_samples_leaf": randint(1, 9),
	              "criterion": ["gini", "entropy"]}

	# Instantiate a Decision Tree classifier: tree
	tree = DecisionTreeClassifier()

	hyperparametertuning_RandomizedSearchCV(tree,param_dist,X,y)

	print "logisticregression"

	hyperparametertuning_RandomizedSearchCV(logreg,logreg_params,X,y)

	print "Knn"
	hyperparametertuning_RandomizedSearchCV(knn,knn_params,X,y)


	# Hold-out set in practice I: Classification

	print "GridSearchCV in training"
	logreg_params = {'C': c_space, 'penalty': ['l1', 'l2']}
	hyperparametertuning_gridSearchCV(logreg, logreg_params, X_train,y_train)









	
	