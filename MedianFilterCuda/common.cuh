#ifndef BDS_COMMON
#define BDS_COMMON "stuff everybody wants, but probably doesn't need"

/* things that you configure for compilation */
#define IMAGE_SIZE 512

/* compiled things that help */

#include <string>
#include <iostream>
#include <map>
#include <functional>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* stupid simple helper functions */
__device__ inline bool in_bounds ( int x ) { return ( x >= 0 && x < IMAGE_SIZE*IMAGE_SIZE ); }

__device__ inline void set_indices ( int& x, int& y, int& cur_pixel_index )
{
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	cur_pixel_index = IMAGE_SIZE*y + x;
}


/* uncompiled things that help */

#include "inc/helper_image.h"

#endif