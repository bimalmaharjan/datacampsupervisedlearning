# Import numpy and pandas
import numpy as np
import pandas as pd
# Import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def get_data()
# Read the CSV file into a DataFrame: df
	df = pd.read_csv('gapminder.csv')
	

	
	# Create arrays for features and target variable
	y = df.life.values
	X = df.fertility.values

	# Print the dimensions of X and y before reshaping
	print("Dimensions of y before reshaping: {}".format(y.shape))
	print("Dimensions of X before reshaping: {}".format(X.shape))

	# Reshape X and y
	y = y.reshape(-1,1)
	X = X.reshape(-1,1)

	# Print the dimensions of X and y after reshaping
	print("Dimensions of y after reshaping: {}".format(y.shape))
	print("Dimensions of X after reshaping: {}".format(X.shape))
	return df

def regression(df)

	# Create the regressor: reg
	reg = LinearRegression()
	X_fertility = df['fertility']

	# Create the prediction space
	prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

	# Fit the model to the data
	reg.fit(X_fertility,y)

	# Compute predictions over the prediction space: y_pred
	y_pred = reg.predict(prediction_space)

	# Print R^2 
	print(reg.score(X_fertility, y))

	# Plot regression line
	plt.plot(prediction_space, y_pred, color='black', linewidth=3)
	plt.show()



	# Create training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

	# Create the regressor: reg_all
	reg_all = LinearRegression()

	# Fit the regressor to the training data
	reg_all.fit(X_train, y_train )

	# Predict on the test data: y_pred
	y_pred = reg_all.predict(X_test)

	# Compute and print R^2 and RMSE
	print("R^2: {}".format(reg_all.score(X_test, y_test)))
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	print("Root Mean Squared Error: {}".format(rmse))


	cross_validation(X,y)

def cross_validation(X,y)
	
	# Create a linear regression object: reg
	reg = LinearRegression()

	# Compute 5-fold cross-validation scores: cv_scores
	cv_scores = cross_val_score(reg, X,y, cv=5)

	# Print the 5-fold cross-validation scores
	print(cv_scores)

	print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


	# Perform 3-fold CV
	cvscores_3 = cross_val_score(reg, X,y, cv=3)
	print(np.mean(cvscores_3))

	# Perform 10-fold CV
	cvscores_10 = cross_val_score(reg, X,y, cv=10)
	print(np.mean(cvscores_10))




if __name__ == '__main__':
	df = get_data()
	regression(df)