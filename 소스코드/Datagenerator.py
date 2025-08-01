# 필요한 라이브러리를 임포트합니다
from google.colab import drive
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 구글 드라이브를 마운트합니다
drive.mount('/content/gdrive')

# 상수를 정의합니다.
A = 0.1        #A = 0.1
B = 0.05       #B = 0.01
N0 = 100  # N(0)

t_data = []
N_t_data = []

# 해석적 해
tau = 1 / A
eta = (B * N0**2) / (A * N0 + B * N0**2)
t2 = np.linspace(0, 20, 100)
N_t = (A / B) * (eta * np.exp(-t2 / tau)) / (1 - eta * np.exp(-t2 / tau))

t_data.append(t2)
N_t_data.append(N_t)

plt.figure(figsize=(10, 6))
plt.plot(t2, N_t, label='Analytical solution', linestyle='dashed')
plt.title('Plot of N(t)')
plt.xlabel('t')
plt.ylabel('N(t)')
plt.legend()
plt.grid(True)
plt.show()

# 데이터 프레임을 생성하고 엑셀 파일로 저장
#df = pd.DataFrame({"t_data": t_data[0], "N_t_data": N_t_data[0]})
#df.to_excel('/content/gdrive/My Drive/output2.xlsx', index=False)

# 데이터를 담을 DataFrame 생성
data = pd.DataFrame({'t': np.concatenate(t_data), 'N': np.concatenate(N_t_data)})

# 데이터를 메모장 파일로 저장
file_path = '/content/gdrive/MyDrive/Experiment_data_2.txt'
data.to_csv(file_path, sep='\t', index=False, header=False)

print('데이터가 성공적으로 저장되었습니다.')