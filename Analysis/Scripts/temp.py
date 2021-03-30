import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

step = 3

def load(path):
	df = pd.read_csv(path)
	queue = df['Queue_Density'].tolist()[:-1]
	dynamic = df['Dynamic_Density'].tolist()[:-1]
	runtime = int(np.round(df.tail(1)['Frame_Num'].tolist()[0]/1000000))
	result = np.array(list(zip(queue, dynamic)))
	return result, runtime

baseline = load('../Outputs/baseline.txt')[0]
result = load('../Outputs/bonus_test.txt')[0]

maxx = -1
factor = -1

for i in range(1, 2000):
	temp = result*i
	accuracy = np.mean(np.abs(baseline-temp), 0)
	accuracy = np.round(100*(1-(accuracy)), 2)[1]
	if accuracy > maxx:
		factor = i
		maxx = accuracy
print(maxx)

plt.show()