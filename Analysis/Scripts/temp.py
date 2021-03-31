import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os import walk
import numpy as np
import sys

method = '1'
mode = 'abs'
baseline_path = '../Outputs/baseline.txt'
dir_path = '../Outputs/Method' + method

f = []
for (dirpath, dirnames, filenames) in walk(dir_path):
    f.extend(filenames)
    break

def get_info(method):
	if method == '1':
		return ' frames'
	if method == '2':
		return ''
	else:
		return ' threads'

labels = [file[:-4] for file in f if file[-4:] != '.png']
# labels = [label+get_info(method) for label in labels]
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
		std = np.round(np.std(np.abs(baseline-result), 0)[1], 2)
	else:
		accuracy = np.sqrt(np.mean(np.square(baseline-result), 0))
		std = np.round(np.sqrt(np.std(np.square(baseline-result), 0))[1], 2)
	accuracy = np.round(100*(1-(accuracy)), 2)
	stats.append([accuracy[0], accuracy[1], std, np.round((accuracy[0]+accuracy[1])/2, 2), int(runtime)])

stats = [stats[-1], stats[3], stats[2], stats[1], stats[0], stats[-2]]
print(stats)
stats = np.array(stats)
labels[-1] = '1'
labels = [int(label) for label in labels]
labels = [labels[-1], labels[3], labels[2], labels[1], labels[0], labels[-2]]
print(labels)
plt.figure(figsize=(10, 6))
plt.plot(labels, stats[:, -1])
plt.ylabel('Run Time (in seconds)')
plt.xlabel('Frames to skip')
plt.legend()
plt.title('Method 1')
# plt.show()
plt.savefig('../runtime.png')