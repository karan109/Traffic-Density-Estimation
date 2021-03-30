import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os import walk
import numpy as np
import sys

method = sys.argv[1]
mode = sys.argv[2]
baseline_path = '../Analysis/Outputs/baseline.txt'
dir_path = '../Analysis/Outputs/Method' + method

f = []
for (dirpath, dirnames, filenames) in walk(dir_path):
    f.extend(filenames)
    break

def get_info(method):
	if method == '1':
		return ' frames'
	if method == '2':
		return ''
	if method == '5':
		return ''
	else:
		return ' threads'

labels = [file[:-4] for file in f if file[-4:] != '.png']
labels = [label+get_info(method) for label in labels]
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
	accuracy = np.round(100*(1-(accuracy)), 2)
	stats.append([accuracy[0], accuracy[1], runtime])


stats = np.array(stats)

# Queue Density
colors = iter(cm.rainbow(np.linspace(0, 1, len(stats))))
for i in range(len(stats)):
	plt.scatter([stats[i, 2]], [stats[i, 0]], color=next(colors), label=labels[i])
plt.title('Queue Density Comparison (Method ' + method + ')')
plt.ylabel('Utility')
plt.xlabel('Run Time (in seconds)')
plt.legend()
plt.savefig(dir_path+'/Queue_Density.png')

plt.figure()

# Dynamic Density
colors = iter(cm.rainbow(np.linspace(0, 1, len(stats))))
for i in range(len(stats)):
	plt.scatter([stats[i, 2]], [stats[i, 1]], color=next(colors), label=labels[i])
plt.title('Dynamic Density Comparison (Method ' + method + ')')
plt.ylabel('Utility')
plt.xlabel('Run Time (in seconds)')
plt.legend()
plt.savefig(dir_path+'/Dynamic_Density.png')

plt.figure()

# Average
colors = iter(cm.rainbow(np.linspace(0, 1, len(stats))))
for i in range(len(stats)):
	plt.scatter([stats[i, 2]], [np.round((stats[i, 1]+stats[i, 0])/2, 2)], color=next(colors), label=labels[i])
plt.title('Average Utility Comparison (Method ' + method + ')')
plt.ylabel('Utility')
plt.xlabel('Run Time (in seconds)')
plt.legend()
plt.savefig(dir_path+'/Average.png')