import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os import walk
import numpy as np
import sys

step = 3

if len(sys.argv) == 1:
	df = pd.read_csv('../Analysis/Outputs/baseline.txt')
	y1 = list(df['Queue_Density'])[::step]
	y2 = list(df['Dynamic_Density'])[::step]
	x = list(df['Frame_Num'])[::step]
	x = [point/15 for point in x]
	plt.figure(figsize=(10, 6))
	plt.plot(x, y1, label='Queue Density')
	plt.plot(x, y2, label='Dynamic Density')
	plt.ylabel('Density')
	plt.xlabel('Time (in seconds)')
	plt.legend()
	plt.title('Baseline')
	plt.savefig('../Analysis/Outputs/Baseline.png')
else:
	method = sys.argv[1]
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
		else:
			return ' threads'

	labels = [file[:-4] for file in f if file[-4:] != '.png']
	labels = [label+get_info(method) for label in labels]

	paths = [dir_path+'/'+file for file in f if file[-4:] != '.png']
	outputs = [dir_path + '/Plots/' + file[:-4] + '.png' for file in f if file[-4:] != '.png']
	i = 0
	for path in paths:
		df = pd.read_csv(path)
		y1 = list(df['Queue_Density'])[::step]
		y2 = list(df['Dynamic_Density'])[::step]
		x = list(df['Frame_Num'])[::step]
		x = [point/15 for point in x]
		plt.figure(figsize=(10, 6))
		plt.plot(x, y1, label='Queue Density')
		plt.plot(x, y2, label='Dynamic Density')
		plt.ylabel('Density')
		plt.xlabel('Time (in seconds)')
		plt.legend()
		plt.title(labels[i])
		plt.savefig(outputs[i])
		i += 1