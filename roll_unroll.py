
import numpy as np

def roll(array,k,n):
   mat = np.ones([n,k])
   for  i in range(n):
      mat[i] = array[i*k:(i*k)+k]

   return mat


def unroll(data_mat):
    n, d = data_mat.shape
    data_vec = np.ndarray(shape=(n*d,), dtype=np.float32)
    for i in range(n):
        data_vec[i*d: (i+1)*d] = data_mat[i]

    return data_vec
 