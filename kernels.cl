
__kernel void compute_weights(__global float *gray, 
                __global int *ngbrs_gpu,
                __global float *weight_vec, const int k, const float scale)


{
    int n, i, j;
    n = get_global_size(0);
    i = get_global_id(0);

    for (j = 0; j < k; ++ j)
    {
               weight_vec[i*k+j] = 1.0f / (1.0f + ( (gray[i]-gray[ngbrs_gpu[i*k+j]])*(gray[i]-gray[ngbrs_gpu[i*k+j]]) / scale )); 
                
               
    }

}


