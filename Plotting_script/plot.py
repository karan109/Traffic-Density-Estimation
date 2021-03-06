import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('../Outputs/user_out.txt')
y1 = list(df['Queue_Density'])
y2 = list(df['Dynamic_Density'])
x = list(df['Frame_Num'])
x = [point/15 for point in x]
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='Queue Density')
plt.plot(x, y2, label='Dynamic Density')
plt.ylabel('Density')
plt.xlabel('Time (in seconds)')
plt.legend()
plt.savefig('../Outputs/user_graph.png')
plt.show()