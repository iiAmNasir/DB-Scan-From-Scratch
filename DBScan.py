# Import Required Libraries
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
import queue
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from pandas import DataFrame
import scipy.io as spio

#Here I have make a dataset of 200 sample and then it is saved to csv file.
#Finally i add 5 outliers to the dataset manually...

"""
X,Y = make_moons(n_samples=200, noise=0.2)

df = pd.DataFrame(X, columns = ['X1','X2'])

df1 = pd.DataFrame(Y, columns = ['Y'])

df['Y']=df1['Y']

df.to_csv("Moon.csv", index=False)
"""



#In previous step I have made a dataset.Now its time to import this data and perform
#some computation
#Load Data

df = pd.read_csv (r'Moon.csv')
Xdf=df.drop(['Y'], axis=1)
X=Xdf.to_numpy()
# Standardize features to zero mean and unit variance.
X = StandardScaler().fit_transform(X)


# Here I define labels for different groups
outliers = 0
not_assigned = 0
core = -1
edge = -2

# function to find all neigbor points in radius
def FindNeighbor(train_data, PId, Radius):
    pts = []
    for i in range(len(train_data)):
        # L2 Norm is used for finding Eucledean distance
        if np.linalg.norm(train_data[i] - train_data[PId]) <= Radius:
            pts.append(i)
    return pts



def dbscan(train_data, Eps, MinPt):

    ptlabel = [not_assigned] * len(train_data)
    ptcount = []
    # define and initialize list for storing core and non core points
    corepts = []
    notcorepts = []

    # Find  neighbor points  of all point
    for i in range(len(train_data)):
        ptcount.append(FindNeighbor(X, i, Eps))

    # Find all core point, edge point and outliers
    for i in range(len(ptcount)):
        if (len(ptcount[i]) >= MinPt):
            ptlabel[i] = core
            corepts.append(i)
        else:
            notcorepts.append(i)

    for i in notcorepts:
        for k in ptcount[i]:
            if k in corepts:
                ptlabel[i] = edge
                break

    #Adding points to cluster
    cls = 1

    for i in range(len(ptlabel)):
        q = queue.Queue()
        if (ptlabel[i] == core):
            ptlabel[i] = cls
            for j in ptcount[i]:
                if (ptlabel[j] == core):
                    q.put(j)
                    ptlabel[j] = cls
                elif (ptlabel[j] == edge):
                    ptlabel[j] = cls
            # When all points are checked in the queue then stop
            while not q.empty():
                neighbors = ptcount[q.get()]
                for k in neighbors:
                    if (ptlabel[k] == core):
                        ptlabel[k] = cls
                        q.put(k)
                    if (ptlabel[k] == edge):
                        ptlabel[k] = cls
            # jump over to next cluster
            cls = cls + 1

    return ptlabel, cls


# Function to plot the final clusters
def plotClusters(train_data, clusterRes, clusterNum):
    nPoints = len(train_data)
    scatterColors = ['black', 'green', 'brown', 'red', 'purple', 'orange', 'yellow']
    for i in range(clusterNum):
        if (i == 0):
            # Plot all outliers as blue
            color = 'blue'
        else:
            color = scatterColors[i % len(scatterColors)]
        x1 = []
        y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(train_data[j, 0])
                y1.append(train_data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='.')



def Test(epsilon,minpoints):

    print('Result on  epsilon = ' + str(epsilon) + ', Minpoints = ' + str(minpoints))
    ptlabel, cl = dbscan(X, epsilon, minpoints)
    plotClusters(X, ptlabel, cl)
    plt.show()
    print('number of cluster found: ' + str(cl - 1))
    outliers = ptlabel.count(0)
    print('numbrer of outliers found: ' + str(outliers))

#Test1 with low Epsilon and high minpoints
#Blue colour represent outliers
Test(0.08,5)
#Test2 with normal Epsilon and normal minpoints
Test(0.1,3)

#Test3 with high Epsilon and low minpoints
Test(0.4,2)
