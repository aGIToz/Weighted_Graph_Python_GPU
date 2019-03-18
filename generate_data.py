#generate the positon and the texture in both the npz format and ply format
import numpy
from open3d import *
import argparse

# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--format", type = int, required=True,
    help="1 is for ply format , else it is npz")
args = vars(ap.parse_args())

# function to convert numpy arrays to ply files
def convert_to_pointCloud(X,Y):
   intensity_f = Y
   my_img_position = X
   my_img_color = np.concatenate((intensity_f,intensity_f,intensity_f),axis=1)
   pcd = PointCloud()
   pcd.points = Vector3dVector(my_img_position)
   pcd.colors = Vector3dVector(my_img_color)
   write_point_cloud("./data/X_Y.ply", pcd)
   return 1


# generate the random data of size one milliion
numpy.random.seed(42)
X = numpy.random.uniform(low=0, high=1,size=(1000000,3))
Y = numpy.random.uniform(low=0, high=1,size=(1000000,1))

#save the data
if args["format"] == 1 :
   numpy.savez("./data/X.npz", position = X)
   numpy.savez("./data/Y.npz", gray = Y)
else :
    tmp = convert_to_pointCloud(X,Y)
    
    

