from pyspark import SparkConf, SparkContext
import sys, operator




def add_tuples(a, b):
    return list(sum(p) for p in zip(a,b))
	
def permutation(row):
	rowPermutation = []
	
	for element in row:
		for e in range(len(row)):
			rowPermutation.append(float(element) * float(row[e]))
			
		
	
	return rowPermutation
	


def main():

	
	input = sys.argv[1]
	output = sys.argv[2]
	
	
	conf = SparkConf().setAppName('Matrix Multiplication')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'
	
	row = sc.textFile(input).map(lambda row : row.split(' ')).cache()
	ncol = len(row.take(1)[0])
	intermediateResult = row.map(permutation).reduce(add_tuples)
	
	outputFile = open(output, 'w') 
	

	
	
	
	result = [intermediateResult[x:x+3] for x in range(0, len(intermediateResult), ncol)]
	
	
	for row in result:
		for element in row:
			outputFile.write(str(element) + ' ')
		outputFile.write('\n')
		
	outputFile.close()
	
	# outputResult = sc.parallelize(result).coalesce(1)
	# outputResult.saveAsTextFile(output)

	
	
	
if __name__ == "__main__":
	main()