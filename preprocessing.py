# Import pandas
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# this function will not work because there is no gapminder data
# this is an example for categorical features


def gaminder():
	# Read 'gapminder.csv' into a DataFrame: df
	df = pd.read_csv('gapminder.csv')

	# Create a boxplot of life expectancy per region
	df.boxplot('life', 'Region', rot=60)

	# Show the plot
	plt.show()

	# Create dummy variables: df_region
	df_region = pd.get_dummies(df)

	# Print the columns of df_region
	print(df_region.columns)

	# Create dummy variables with drop_first=True: df_region
	df_region = pd.get_dummies(df, drop_first=True)

	# Print the new columns of df_region
	print(df_region.columns)

def pipeline_example():

		# Setup the Imputation transformer: imp
	imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

	# Instantiate the SVC classifier: clf
	clf = SVC()

	# Setup the pipeline with the required steps: steps
	steps = [('imputation', imp),
	        ('SVM', clf)]

def pipeline_congressional_votes():

	file_name = 'congressionalvotes1984.csv'
	columns = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
       'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
       'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']
    
	df = pd.read_csv(file_name,sep=',', names=columns)

	# Convert '?' to NaN
	df[df == '?'] = np.nan

	# Setup the pipeline steps: steps
	steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
	        ('SVM', SVC())]

	# Create the pipeline: pipeline
	pipeline = Pipeline(steps)

	# replace y by 1 and n by 0
	df = df.replace('y',1)
	df = df.replace('n',0)	


	y = df['party'].values 
	X = df.drop('party', axis=1).values

	# Create training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

	# Fit the pipeline to the train set
	pipeline.fit(X_train, y_train)

	# Predict the labels of the test set
	y_pred = pipeline.predict(X_test)

	# Compute metrics
	print(classification_report( y_test, y_pred))


if __name__ == '__main__':
	pipeline_congressional_votes()
