import numpy as np
import sympy as sy

M = sy.Matrix([[24294.3, 23763.3, -22564, -27522, 2028.72],
              [23763.3, 37215.3, -26584, -31211, -3183.3],
              [-22564, -26584, 32686.9, 25780.3, -9319.5],
              [-27522, -31211, 25780.3, 45872.7, -12919],
              [2028.72, -3183.3, -9319.5, -12919, 23393.1]])

# Eigen values
lamda = sy.symbols('lamda')
p = M.charpoly(lamda)
eigenVals = sy.solve(p, lamda)
print(eigenVals)

# Eigen vectors

