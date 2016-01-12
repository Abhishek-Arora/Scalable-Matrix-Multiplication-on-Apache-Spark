from pyspark import SparkConf, SparkContext
import sys, operator
from scipy import *
from scipy.sparse import csr_matrix



def createCSRMatrix(input):
	row = []
	col = []
	data = []

	for values in input:
		value = values.split(':')
		row.append(0)
		col.append(int(value[0]))
		data.append(float(value[1]))
		
	return csr_matrix((data,(row,col)), shape=(1,100))
	

def multiplyMatrix(csrMatrix):

    csrTransponse = csrMatrix.transpose(copy=True)

    return (csrTransponse*csrMatrix)
	
def formatOutput(indexValuePairs):
	return ' '.join(map(lambda pair : str(pair[0]) + ':' + str(pair[1]), indexValuePairs))
		

def main():

	
	input = sys.argv[1]
	output = sys.argv[2]
	
	
	conf = SparkConf().setAppName('Sparse Matrix Multiplication')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'
	
	sparseMatrix = sc.textFile(input).map(lambda row : row.split(' ')).map(createCSRMatrix).map(multiplyMatrix).reduce(operator.add)
	outputFile = open(output, 'w')
	
	for row in range(len(sparseMatrix.indptr)-1):
		col = sparseMatrix.indices[sparseMatrix.indptr[row]:sparseMatrix.indptr[row+1]]
		data = sparseMatrix.data[sparseMatrix.indptr[row]:sparseMatrix.indptr[row+1]]
		indexValuePairs = zip(col,data)
		formattedOutput = formatOutput(indexValuePairs)
		outputFile.write(formattedOutput + '\n')
		

	
if __name__ == "__main__":
	main()

