#ifndef BDS_SCS_GPU
#define BDS_SCS_GPU "5/11/2015 Ben Sammons"

#include "common.cuh"

__global__ void sobel_kernel ( unsigned char *in, unsigned char *out )
{
	int x_index, y_index, pixel_index;
	set_indices ( x_index, y_index, pixel_index );

	const int x = x_index;
	const int y = y_index;

	const double kernel_x[ 3 ][ 3 ] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	const double kernel_y[ 3 ][ 3 ] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };

	double magnitude_x = 0.0;
	double magnitude_y = 0.0;
	for ( int i = 0; i < 3; i++ )
	{
		const int x_local = i + x;
		for ( int j = 0; j < 3; j++ )
		{
			const int y_local = j + y;
			const int index = x_local + y_local * IMAGE_SIZE;
			magnitude_x += in[ index ] * kernel_x[ i ][ j ];
			magnitude_y += in[ index ] * kernel_y[ i ][ j ];
		}
	}
	out[ x + y*IMAGE_SIZE ] = sqrt ( magnitude_x*magnitude_x + magnitude_y*magnitude_y );
}

#endif