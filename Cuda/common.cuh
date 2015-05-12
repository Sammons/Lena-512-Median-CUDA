#ifndef BDS_COMMON
#define BDS_COMMON "stuff everybody wants, but probably doesn't need"

/* things that you configure for compilation */
#define IMAGE_SIZE 512

/* compiled things that help */

#include <string>
#include <iostream>
#include <map>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* stupid simple helper functions */
__device__ inline bool in_bounds ( int x ) { return ( x >= 0 && x < IMAGE_SIZE*IMAGE_SIZE ); }
inline bool in_bounds_cpu ( int x ) { return ( x >= 0 && x < IMAGE_SIZE*IMAGE_SIZE ); }

__device__ inline void set_indices ( int& x, int& y, int& cur_pixel_index )
{
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	cur_pixel_index = IMAGE_SIZE*y + x;
}
/* timer functions */
 static std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> time_map {};

 inline void start ( std::string timer_name )
{
	time_map[ timer_name ] = std::chrono::high_resolution_clock::now ();
}

 inline double get_time ( std::string timer_name )
{
	auto start = time_map[ timer_name ];
	auto stop = std::chrono::high_resolution_clock::now ();
	auto duration = std::chrono::duration_cast<std::chrono::duration<double> >( stop - start );
	return duration.count ();
}

/* uncompiled things that help */

#include "inc/helper_image.h"

#endif