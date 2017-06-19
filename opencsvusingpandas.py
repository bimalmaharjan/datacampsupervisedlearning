import pandas as pd
import numpy as np


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


if __name__ == '__main__':
 	df = importfile()
 	df = preprocessing(df)
 	numerical_eda(df)
