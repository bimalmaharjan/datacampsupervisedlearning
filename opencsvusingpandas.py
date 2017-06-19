import pandas as pd


def importfile():
	file_name = 'congressionalvotes1984.csv'
	columns = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
       'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
       'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']
    
	df = pd.read_csv(file_name,sep=',', names=columns)
	print df
    


if __name__ == '__main__':
 	importfile()