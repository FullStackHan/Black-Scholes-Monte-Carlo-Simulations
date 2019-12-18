import numpy as np
import matplotlib.pyplot as plt

filename="/home/xugang/BI_demo/cuda-samples_8.0/4_Finance/stockast/opt.csv"
bs_arr = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
paths = bs_arr[:,0:-1]
avg_path =  bs_arr[:,-1]
plt.figure(figsize=(8,5))
plt.plot(paths)
plt.plot(avg_path, linewidth='3', label="avg", color='k')

plt.title('Simulated Path')
plt.ylabel('Price')
plt.xlabel('TimeStep')
plt.show()

