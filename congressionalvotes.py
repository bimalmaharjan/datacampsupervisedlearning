import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def importfile():
	file_name = 'congressionalvotes1984.csv'
	columns = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
       'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
       'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']
    
	df = pd.read_csv(file_name,sep=',', names=columns)
	return df

def preprocessing(df):
	
	# Convert '?' to NaN
	df[df == '?'] = np.nan

	# Print the number of NaNs
	print(df.isnull().sum())

	# Print shape of original DataFrame
	print("Shape of Original DataFrame: {}".format(df.shape))

	# Drop missing values and print shape of new DataFrame
	df = df.dropna()

	# Print shape of new DataFrame
	print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))

	# replace y by 1 and n by 0
	df = df.replace('y',1)
	df = df.replace('n',0)	
	return df

def numerical_eda(df):
	print df.head()
	print df.describe()
	print df.info()  

def visual_eda(df):
	
	# counter plot for education
	plt.figure()
	sns.countplot(x='education', hue='party', data=df, palette='RdBu')
	plt.xticks([0,1], ['No', 'Yes'])
	plt.show()

	# counter plot for satellite
	plt.figure()
	sns.countplot(x='satellite', hue='party', data=df, palette='RdBu')
	plt.xticks([0,1], ['No', 'Yes'])
	plt.show()

	# counter plot for missile
	plt.figure()
	sns.countplot(x='missile', hue='party', data=df, palette='RdBu')
	plt.xticks([0,1], ['No', 'Yes'])
	plt.show()

def classify(df):
	# Create arrays for the features and the response variable
	y = df['party'].values
	X = df.drop('party', axis=1).values
	

	# Create a k-NN classifier with 6 neighbors
	knn = KNeighborsClassifier(n_neighbors = 6)

	# Fit the classifier to the data
	knn.fit(X,y)

	# Predict the labels for the training data X
	y_pred = knn.predict(X)

	X_new = pd.DataFrame([[ 0.23810678,  0.15188048,  0.64483788,  0.50831858,  0.90189808,
         0.28908815,  0.96460336,  0.97180874,  0.92409025,  0.58541324,
         0.58395216,  0.78974121,  0.24326996,  0.18586739,  0.65712813,
         0.82900472]])
	
	# Predict and print the label for the new data point X_new
	new_prediction = knn.predict(X_new)
	print("Prediction: {}".format(new_prediction))


	# Split into training and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

	# Create a k-NN classifier with 7 neighbors: knn
	knn = KNeighborsClassifier(n_neighbors=6)

	# Fit the classifier to the training data
	knn.fit(X_train,y_train)

	# Print the accuracy
	print(knn.score(X_test, y_test))





if __name__ == '__main__':
 	df = importfile()
 	df = preprocessing(df)
 	# numerical_eda(df)
 	# visual_eda(df)
 	classify(df)

