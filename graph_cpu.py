from time import time
import numpy as np
import argparse
from scipy import  spatial

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--position", type = str, required=True,
    help="path to 3d cordinates, npz format")
ap.add_argument("-t", "--texture", type = str, required=True,
    help="path to the texture, for the moment just for gray scale, npz format")
ap.add_argument("-k", "--k", type = int,  required=True,
    help="number of nns")
args = vars(ap.parse_args())


# get the position vector               
position = np.load(args["position"])    
position = position['position']
position = position.astype('float32')
N, d = np.shape(position)
k = args["k"]

# otimize your data,  a not necessarily required step
position = position- np.kron(np.ones((N, 1)),np.mean(position, axis=0))
bounding_radius = 0.5 * np.linalg.norm(np.amax(position, axis=0) - np.amin(position, axis=0), 2)
temp = np.power(N, 1. / float(min(d, 3))) / 10.
position *= temp / bounding_radius

# build the tree and find the neighbors
start = time()
print("process started, now wait ...")
kdt = spatial.KDTree(position)
D, NN = kdt.query(position, k=(k + 1),p=2)

# get the intensity values
gray = np.load(args["texture"])                 
gray = gray['gray']
gray = gray[0:]
gray = 255*gray
gray = np.reshape(gray, gray.size)
k = args["k"]
scale = 1

# from the neighbors make the graph
spj = np.zeros((N * k))
my_spv = np.zeros((N * k))
for i in range(N):
    spj[i * k:(i + 1) * k] = NN[i, 1:]
    my_spv[i * k:(i + 1) * k] = 1/(1+(((gray[spj[i * k:(i + 1) * k].astype('int')] - gray[i])**2) /float(scale)))
    #my_spv[i * k:(i + 1) * k] = 1/(1+(((my_Y[spj[i * k:(i + 1) * k].astype('int')] - my_Y[i])**2) /float(self.sigma))) 
end = time() - start 
print('time taken for building the graph on cpu :',end)

# save your graph
np.savez("./graphs/graph_cpu.npz",weights=my_spv, ngbrs=spj)

 
