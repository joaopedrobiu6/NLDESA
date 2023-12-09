import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

def scatter(features,labels):
    x = features[:,0]
    y = features[:,1]

    z = labels[:]

    plt.scatter(x, y, c=np.where(z == 1, 'gold', 'indigo'),s=1)

    plt.show()
    

data = np.load('data/training_set.npz')

features = data['features']
labels = data['labels']

scatter(features,labels)