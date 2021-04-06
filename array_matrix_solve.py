import numpy as np
import matplotlib.pyplot as plt

# Calculate 2x2 matrix determinate
def determinant(matrix):
  """
  This will take a 2x2 NumPy array and return its determinant (as a float)
  """
  if len(matrix.shape) != 2:
    raise ValueError('Array is not a matrix!')

  for dim in matrix.shape:
    if dim != 2:
      raise ValueError('Matrix is not 2x2')
  
  det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

  return det


# Calculate inverse of 2x2 matrix
def matrix_inverse(matrix):
  """
  This will take a 2x2 NumPy array and will return its inverse if it exists.
  """
  matrix_det = determinant(matrix)
  if matrix_det == 0:
    raise ValueError('Matrix has zero determinant')
  
  inverse_matrix = np.array([
    [matrix[1][1], -matrix[0][1]],
    [-matrix[1][0], matrix[0][0]]
  ])
  inverse_matrix = inverse_matrix / matrix_det

  return inverse_matrix

# Computes solution of 2 variable
## input will be 2x2 array, 2x1 array, output = 2x1 array
def array_matrix_solve(matrix, vector):
  """
  Takes a 2x2 and 2x1 NumPy array and return a 2x2 numpy array equal to the matrix product of the first array with the second array
  """
  if len(vector) != 2:
    raise ValueError('Vector has wrong dimensions')
  
  if (vector.shape[1] != 1) and (vector.shape[0] != 2):
    raise ValueError('Vector has wrong length')

  inv_matrix = matrix_inverse(matrix)

  solution = np.matmul(inv_matrix, vector)

  return solution

# Plot the solution
## input = 2x2 array, 2x1 array
### Convert matrix, column vec into slope-intercept,
### Determine x-range, compute array of y-values
### plot the two
### plot dot at intersection, label solution
def plot_solution(matrix, vector):
  slopes = -matrix[:,0] / matrix[:,1]
  intercepts = vector[:,0] / matrix[:,1]

  x_values = np.linspace(-5,5)

  for slope, intercept in zip(slopes, intercepts):
    y_values = slope * x_values + intercept
    plt.plot(x_values, y_values)

  intersection = array_matrix_solve(matrix, vector)
  coord_label = f"({intersection[0][0]}, {intersection[1][0]})"

  plt.scatter(intersection[0][0], intersection[1][0])
  plt.text(intersection[0][0], intersection[1][0], coord_label)

  plt.show()