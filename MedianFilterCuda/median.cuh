#ifndef BDS_GPU_SCM
#define BDS_GPU_SCM "5/11/2015 Ben Sammons"

#include "common.cuh"
#include <stdio.h>

/* templatized single channel median kernel, please use with a good odd number */
template<int filtersize>
__global__ void median_kernel ( unsigned char *in, unsigned char *out )
{
	/* for convenience */
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int index = IMAGE_SIZE*y + x;

	const int top_left = index - IMAGE_SIZE*(filtersize/2) - filtersize / 2;
	unsigned char * neighbors = ( unsigned char* )malloc ( sizeof ( unsigned char )*filtersize*filtersize );
	/* split the empty neighbors evenly */
#pragma unroll
	for (register int i = 0; i < filtersize*filtersize; ++i )
		neighbors[ i ] = i % 2 == 0 ? 255 : 0;

	/* set neighbors */
	for ( register int i = 0; i < filtersize; i++ )
	{
		const int row_start = top_left + IMAGE_SIZE*i;
#pragma unroll
		for (register int j = 0; j < filtersize; j++ )
		{
			const int cur = row_start + j;
			const int negh = ( i + 1 )*( j + 1 ) - 1;
			neighbors[ negh ] = in_bounds ( cur ) ? in[ cur ] : neighbors[ negh ];
		}
	}
	

	/* bubble sort */
	register bool swap_happened = true;
	while ( swap_happened )
	{
		swap_happened = false;
#pragma unroll
		for ( int k = 1; k < filtersize*filtersize; ++k )
		{
			const int i = k-1;
			const int tmp = neighbors[ k ];
			if ( tmp > neighbors[ i ] )
			{
				swap_happened = true;
				neighbors[ k ] = neighbors[ i ];
				neighbors[ i ] = tmp;
			}
		}
	}

	/* set */
	out[ index ] = neighbors[ filtersize*filtersize / 2 ];
	free ( neighbors );
}

/* explicit instantiation of median kernels */
template __global__ void median_kernel<3> ( unsigned char *in, unsigned char *out );
template __global__ void median_kernel<5> ( unsigned char *in, unsigned char *out );
template __global__ void median_kernel<7> ( unsigned char *in, unsigned char *out );
template __global__ void median_kernel<9> ( unsigned char *in, unsigned char *out );
template __global__ void median_kernel<11> ( unsigned char *in, unsigned char *out );
template __global__ void median_kernel<15> ( unsigned char *in, unsigned char *out );

typedef void ( *median_kernel_t )( unsigned char *in, unsigned char *out );

/* for conveniently fetching the median kernel we are interested in executing */
static std::map<std::string, median_kernel_t> kernel_map = {
	{ "3", median_kernel  < 3 > },
	{ "5", median_kernel  < 5 > },
	{ "7", median_kernel  < 7 > },
	{ "9", median_kernel  < 9 > },
	{ "11", median_kernel < 11 > },
	{ "15", median_kernel < 15 > }
};

/* wrapper for extracting function given a string with filter size */
median_kernel_t get_median_kernel ( std::string filter_size_str )
{
	if ( kernel_map.find ( filter_size_str ) == kernel_map.end () )
	{
		std::cout << "No median kernel found with requested dimension, defaulting to 3x3" << std::endl;
		return median_kernel < 3 >;
	}
	return kernel_map[ filter_size_str ];
}

#endif