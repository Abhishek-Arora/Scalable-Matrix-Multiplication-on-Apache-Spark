# Scalable-Matrix-Multiplication-on-Apache-Spark
This project aims at providing a scalable approach to matrix multiplication, which is one of the most used step in machine learning.

##The Problem in Scalable Machine Learning

At the first sight, scalable machine learning (ML) seems to be an easy thing to do because Spark has already provided scalable data processing. That is, if we could re-implement existing ML algorithms using Spark, the ML algorithms would inherit the scalability feature (i.e., scaling out to 100+ machines and dealing with petabytes of data) from Spark for free.

However, the challenging part is that for an ML algorithms that works well on a single machine, it does not mean that the algorithm can be easily extended to the Spark programming framework. Furthermore, to make the algorithm run fast in a distributed environment, we need to carefully select our design choices (e.g., broadcast or not, dense or sparse representation).

##Dense Matrix Multiplication

* matrix_multiply.py

In this file, we will deal with a case that the matrix  AA  has a big  n  and a small  b  (e.g.,  n=10^9,b=10). In this case, the matrix can not be stored in a single machine, so you have to distribute the storage.

##Approach

Insted of using the standard inner product way, this program expresses matrix multiplication in the form of outer product.

###Input

The input for this program is in the file matrix_multiply.txt which is located in the directory called 'data'.

###Output

The output is 10x10 matrix which is saved in the output file.

##Command to Run

spark-submit --master <MASTER> matrix_multiply.py path_for_matrix_data path_for_output_result


##Sparse Matrix Multiplication

* matrix_multiply_sparse.py

As mentioned in the beginning of this section, to develop an efficient distributed algorithm, we need to carefully select our design choices (e.g., broadcast or not, dense or sparse representation). Next, you will see how to use sparse representation to improve the performance of matrix multiplication.


Suppose you want to compute  A_transpose*A  as before. But unlike the dense matrix multiplication, here the matrix  A  is very sparse, where most of the elements in the matrix are zero. If you use the same algorithm as before, the computation cost will be  O(n∗d^2)O(n∗d2) . We will try to reduce the computation cost to  O(n∗s^2)O(n∗s^2)  via sparse representation, where  s  is the number of non-zero elements in each row.

###Dataset

The dataset for the sparse matrix can be found in the 'data' directory

The file has  n  lines, and each line represents a row of the matrix. The row is a d = 100 dimentional vector. The vector is very sparse, which is in the format of
index1:value1 index2:value2 index3:value3 ...

where index is the position of a non-zero element, and value is the non-zero element. Note that index starts from zero, so it is in the range of [0, 99]. For example, "0:0.1 2:0.5 99:0.9" represents the vector of "[0.1, 0, 0.5, 0, 0, ... , 0, 0.9]".

###Output

We will compute A_Transpose * A and output the result as a file. The result will be a 100x100 matrix.

##Command to Run

spark-submit --master <MASTER> matrix_multiply_sparse.py path_for_sparse_matrix_data path_for_output_result
