import pandas as pd
import matplotlib.pyplot as plt
from os import walk
import numpy as np

method = '4'
mode = 'rms'
baseline_path = '../Outputs/baseline.txt'
dir_path = '../Outputs/Method' + method

f = []
for (dirpath, dirnames, filenames) in walk(dir_path):
    f.extend(filenames)
    break
labels = [file[:-4] for file in f if file[-4:] != '.png']
labels.append('Baseline')
paths = [dir_path+'/'+file for file in f if file[-4:] != '.png']
paths.append(baseline_path)

def load(path):
	df = pd.read_csv(path)
	queue = df['Queue_Density'].tolist()[:-1]
	dynamic = df['Dynamic_Density'].tolist()[:-1]
	runtime = int(np.round(df.tail(1)['Frame_Num'].tolist()[0]/1000000))
	result = np.array(list(zip(queue, dynamic)))
	return result, runtime

baseline = load(baseline_path)[0]

stats = []
for path in paths:
	result, runtime = load(path)
	if(mode == 'abs'):
		accuracy = np.mean(np.abs(baseline-result), 0)
	else:
		accuracy = np.sqrt(np.mean(np.square(baseline-result), 0))
	accuracy = np.round(100*(1-np.tanh(accuracy)), 2)
	stats.append([accuracy[0], accuracy[1], runtime])


stats = np.array(stats)

# Queue Density
plt.scatter(stats[:, 2], stats[:, 0])
for i in range(len(stats)):
	plt.text(stats[i, 2], stats[i, 0], labels[i])
plt.title('Queue Density Comparison (Method ' + method + ')')
plt.ylabel('Utility')
plt.xlabel('Run Time (in seconds)')
plt.savefig(dir_path+'/Queue_Density.png')
# plt.show()

plt.figure()

# Dynamic Density
plt.scatter(stats[:, 2], stats[:, 1])
for i in range(len(stats)):
	plt.text(stats[i, 2], stats[i, 1], labels[i])
plt.title('Dynamic Density Comparison (Method ' + method + ')')
plt.ylabel('Utility')
plt.xlabel('Run Time (in seconds)')
plt.savefig(dir_path+'/Dynamic_Density.png')
# plt.show()