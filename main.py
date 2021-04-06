from array_matrix_solve import plot_solution
import numpy as np
def main():

  mat = np. array([[1,2], [4,5]])
  vector = np.array([[3],[6]])
  plot_solution(mat, vector)

  

if __name__ == '__main__':
  main()