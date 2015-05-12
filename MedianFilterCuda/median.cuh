#ifndef BDS_GPU_SCM
#define BDS_GPU_SCM "5/11/2015 Ben Sammons"

#include "common.cuh"

/* templatized single channel median kernel, please use with a good odd number */
template<int FILTER_WIDTH>
__global__ void median_kernel ( unsigned char *in, unsigned char *out )
{
	/* for convenience */
	const int FILTER_WIDTH_SQUARED = FILTER_WIDTH * FILTER_WIDTH;

	/* get the thread local values for where we are */
	int x_index, y_index, pixel_index;
	set_indices ( x_index, y_index, pixel_index );

	/* const those values for the compiler */
	const int x = x_index;
	const int y = y_index;
	const int index = pixel_index;

	/* calculate top left of median filter mask */
	const int top_left = index - IMAGE_SIZE - ( FILTER_WIDTH >> 1 );

	/* allocate space for the neighborlist */
	unsigned char * neighbors = ( unsigned char* )malloc ( sizeof ( unsigned char )*FILTER_WIDTH_SQUARED );

	/* every pixel lies on one absolute edge 0, or 255, so real pixels are most likely to be the median */
#pragma unroll
	for ( int i = 0; i < FILTER_WIDTH_SQUARED; ++i )
		neighbors[ i ] = i % 2 == 0 ? 255 : 0;

	/* set neighbors */
#pragma unroll
	for ( int i = 0; i < FILTER_WIDTH; i++ )
	{
		const int row_start = top_left + IMAGE_SIZE*i;
#pragma unroll
		for ( int j = 0; j < FILTER_WIDTH; j++ )
		{
			if ( in_bounds ( row_start + j ) )
				neighbors[ ( i + 1 )*( j + 1 ) - 1 ] = in[ row_start + j ];
		}
	}

	/* bubble sort */
	bool swap_happened = false;
	do
	{
		for ( int i = 1; i < FILTER_WIDTH_SQUARED; ++i )
			if ( neighbors[ i ] > neighbors[ i - 1 ] )
			{
				int tmp = neighbors[ i ];
				neighbors[ i ] = neighbors[ i + 1 ];
				neighbors[ i + 1 ] = tmp;
			}
	} while ( swap_happened );

	/* set pixel to be median */
	out[ index ] = neighbors[ FILTER_WIDTH_SQUARED >> 1 ];
	free ( neighbors );
}

/* explicit instantiation of median kernels */
template
__global__ void median_kernel<3> ( unsigned char *in, unsigned char *out );
template
__global__ void median_kernel<7> ( unsigned char *in, unsigned char *out );
template
__global__ void median_kernel<11> ( unsigned char *in, unsigned char *out );
template
__global__ void median_kernel<15> ( unsigned char *in, unsigned char *out );

typedef void ( *median_kernel_t )( unsigned char *in, unsigned char *out );

/* for conveniently fetching the median kernel we are interested in executing */
static std::map<std::string, median_kernel_t> kernel_map = {
	{ "3", median_kernel < 3 > },
	{ "7", median_kernel < 7 > },
	{ "11", median_kernel < 11 > },
	{ "15", median_kernel < 15 > }
};

/* wrapper for extracting function given a string with filter size */
const median_kernel_t get_median_kernel ( std::string filter_size_str )
{
	if ( kernel_map.find ( filter_size_str ) == kernel_map.end () )
	{
		std::cout << "No median kernel found with requested dimension, defaulting to 3x3" << std::endl;
		return median_kernel < 3 >;
	}
	return kernel_map[ filter_size_str ];
}

#endif