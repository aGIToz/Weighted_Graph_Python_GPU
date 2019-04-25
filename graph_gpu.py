# finding the neighbors part on gpu

from time import time
import numpy 
from bufferkdtree.neighbors import NearestNeighbors
import roll_unroll
import pyopencl as cl
import pyopencl.array
import numpy as np
import argparse


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--position", type = str, required=True,
    help="path to 3d cordinates, npz format")
ap.add_argument("-t", "--texture", type = str, required=True,
    help="path to the texture, for the moment just for gray scale, npz format")
ap.add_argument("-k", "--k", type = int,  required=True, help="number of nns")
ap.add_argument("-s", "--s", type = float,  required=True, help="scale value in the graph")
args = vars(ap.parse_args())
    
position = numpy.load(args["position"])    
position = position['position']
position = position.astype('float32')
print(position.shape)


# find the nearest neighbors on the gpu
start = time()
nbrs = NearestNeighbors(n_neighbors=args["k"]+1, algorithm="buffer_kd_tree", tree_depth=9, plat_dev_ids={0:[0]})    # use the arg parser here
nbrs.fit(position)
dists, inds = nbrs.kneighbors(position)  


# now build the graph using those nns using gpu
platform = cl.get_platforms()[0]
print(platform)

device = platform.get_devices()[0]
print(device)
 
context = cl.Context([device])
print(context)

program = cl.Program(context, open("kernels.cl").read()).build()
print(program)

queue = cl.CommandQueue(context)
print(queue)

 # define the input here which is the ndbrs gpu
ngbrs_gpu = inds
ngbrs_gpu = ngbrs_gpu[0:,1:]
ngbrs_gpu = roll_unroll.unroll(ngbrs_gpu)
ngbrs_gpu = ngbrs_gpu.astype('int32')

 # define the second input here which is the gray levels
gray = np.load(args["texture"])                 
gray = gray['texture']
gray = gray[0:]
gray = 255*gray
gray = gray.astype('float32')
 
k = args["k"]
n =len(gray)
scale = args["s"]

 # create the buffers on the device, intensity, nbgrs, weights
mem_flags = cl.mem_flags
ngbrs_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,hostbuf=ngbrs_gpu)
intensity_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=gray)
weight_vec = np.ndarray(shape=(n*k,), dtype=np.float32)
weight_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, weight_vec.nbytes)

 # run the kernel to compute the weights
program.compute_weights(queue, (n,), None, intensity_buf, ngbrs_buf, weight_buf, np.int32(k), np.float32(scale))

queue.finish()

 # copy the weihts to the host memory
cl.enqueue_copy(queue, weight_vec, weight_buf)
end = time() - start

print('total time taken by the gpu python:', end)
# save the graph
np.savez("./graphs/graph_gpu.npz",weights=weight_vec, ngbrs=ngbrs_gpu)
