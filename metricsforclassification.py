import pandas as pd 
import numpy as numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

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
	return X_train, X_test, y_train, y_test


def classifier(df,clf, X_train, X_test, y_train, y_test):

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


def plot_roc_curve(fpr, tpr):
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.show()



if __name__ == '__main__':
	df = import_file()

	# split test train data
	X_train, X_test, y_train, y_test = get_train_test_data(df)

	# instantiate KNN
	knn = KNeighborsClassifier(n_neighbors= 6)

	# instantiate logisticregression
	logreg = LogisticRegression()

	# classify using knn
	print "knn"
	classifier(df,knn, X_train, X_test, y_train, y_test)

	print "LogisticRegression"
	# classify using logistic regression
	classifier(df, logreg, X_train, X_test, y_train, y_test)


	
	