import numpy as np
import matplotlib.pyplot as plt

#get 2x2 matrix det
def determinant(matrix):
  """
  This will take a 2x2 NumPy array and return its determinant as a float
  """

  if len(matrix.shape) != 2:
    raise ValueError('Array is not a matrix')

  for dim in matrix.shape:
    if dim != 2:
      raise ValueError('Matrix is not 2x2')


  
  det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

  return det





#get inverse of 2x2 matrix
def matrix_inverse(matrix):
  """
  This will take a 2x2 NumPy array and will return its inverse if it exists
  """
  matrix_det = determinant(matrix)
  if matrix_det == 0:
    raise ValueError('Matrix has zero derterminant')


  inverse_matrix = np.array([
    [matrix[1][1], -matrix[0][1]],
    [-matrix[1][0], matrix[0][0]]
  ])
  inverse_matrix = inverse_matrix / matrix_det

  return inverse_matrix

#get solution of 2 variable system
##input 2x2 and 2x1 array, output 2x1 max
def array_matrix_solve(matrix, vector):
  """
  Takes a 2x2 and 2x1 NumPy array and returns a 2x2 numpy array equal to the matrix product of the first array with the second array
  """
  if len(vector) != 2:
    raise ValueError('Vector has wrong dimensions')

  if (vector.shape[1] != 1) and (vector.shape[0] != 2):
    raise ValueError('Vector has wrong length')


  inv_matrix = matrix_inverse(matrix)

  solution = np.matmul(inv_matrix, vector)

  return solution

#plot solution
#convert arrays into slope and intercept of 2 lines
#x range and y range
#dot at intercept

def plot_solution(matrix, vector):
  slopes = -matrix[:,0] / matrix[:,1]
  intercepts = vector[:,0] / matrix[:,1]
  

  x_values = np.linspace(-5,5)
  #y_values = slope * x_values + intercept

  for slope, intercept in zip(slopes, intercepts):
    y_values = slope * x_values + intercept
    plt.plot(x_values, y_values)
  
intersection = array_matrix_solve(matrix, vector)

coord_label = f"({intersection[0][0]}, {intersection[1][0]})"

plt.scatter(intersection[0][0], intersection[1][0])
plt.text(intersection[0][0], intersection[1][0], coord_label)

plt.show()