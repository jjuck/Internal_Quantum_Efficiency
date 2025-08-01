import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pandas as pd

# 텍스트 파일 경로
file_path = '/content/drive/MyDrive/Experiment_data_2.txt'

# 텍스트 파일을 데이터프레임으로 읽기
df = pd.read_csv(file_path, delimiter='\t', header=None)

# 첫 번째 열과 두 번째 열의 데이터 추출
col1 = df[0].values
col2 = df[1].values

col1_array = np.array(col1)
col2_array = np.array(col2)

t_data = col1_array
N_t_data = col2_array

# Differential equation
def dN_dt(t, N, A, B):
    return -A * N - B * N**2

# RK4 integration
def rk4(t, h, y, A, B):
    k1 = h * dN_dt(t, y, A, B)
    k2 = h * dN_dt(t + 0.5 * h, y + 0.5 * k1, A, B)
    k3 = h * dN_dt(t + 0.5 * h, y + 0.5 * k2, A, B)
    k4 = h * dN_dt(t + h, y + k3, A, B)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Model to fit the data
def model(t, y0, A, B):
    h = t[1] - t[0] # Assume uniform spacing
    y = np.empty_like(t)
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = rk4(t[i-1], h, y[i-1], A, B)
    return y

# Objective function for least squares
def fun(params, t, y):
    return model(t, params[0], params[1], params[2]) - y

# Initial guess
params0 = [N_t_data[0], 1, 0]

# Solve using least squares
res = least_squares(fun, params0, args=(t_data, N_t_data), method='lm')

# Print the parameters
print(f"A = {res.x[1]}, B = {res.x[2]}")

# Plot
plt.figure()
plt.plot(t_data, N_t_data, 'ko', label='Data')
plt.plot(t_data, model(t_data, res.x[0], res.x[1], res.x[2]), 'r-', label='Fit')
plt.xlabel('t')
plt.ylabel('N(t)')
plt.legend()
plt.show()