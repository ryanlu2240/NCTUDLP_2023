import matplotlib.pyplot as plt
import csv
import numpy as np


x = []
y = []
with open('training_data.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
    
    for row in plots:
        x.append(int(row[0]))
        y.append(float(row[1]))

plt.plot(x, y)
plt.xlabel('epoch')
plt.ylabel('score')
plt.yticks(np.arange(5000, 70000, step=5000))
plt.savefig('output.png')
