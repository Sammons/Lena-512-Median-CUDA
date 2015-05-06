
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "inc/helper_image.h"
#include "thrust\device_vector.h"
#include "thrust\sort.h"
#include <stdio.h>
#include <iostream>
#include <string>

#define im_size 512

__device__ inline bool in_bounds ( int x ) { return ( x >= 0 && x < im_size*im_size ); }

__global__ void sobel_kernel ( unsigned char *in, unsigned char *out )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	double kernel_x[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	double kernel_y[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };

	double magnitude_x = 0.0;
	double magnitude_y = 0.0;
	for ( int i = 0; i < 3; i++ )
	{
		for ( int j = 0; j < 3; j++ )
		{
			int x_local = i + x;
			int y_local = j + y;
			int index = x_local + y_local * im_size;
			magnitude_x += in[ index ] * kernel_x[ i ][ j ];
			magnitude_y += in[ index ] * kernel_y[ i ][ j ];
		}
	}
	out[ x + y*im_size ] = sqrt( magnitude_x*magnitude_x + magnitude_y*magnitude_y );
}

__global__ void median_kernel(unsigned char *in, unsigned char *out, int filtersize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = im_size*y + x;

	int top_left = index - im_size - filtersize / 2;
	unsigned char * neighbors = (unsigned char*)malloc( sizeof(unsigned char)*filtersize*filtersize);
	/* hack to split the empty neighbors evenly */
	for ( int i = 0; i < filtersize*filtersize; ++i )
		neighbors[ i ] = i % 2 == 0 ? 255 : 0;

	/* set neighbors */
	for ( int i = 0; i < filtersize; i++ )
	{
		const int row_start = top_left + im_size*i;
		for ( int j = 0; j < filtersize; j++ )
		{
			if ( in_bounds ( row_start + j ) )
				neighbors[ ( i + 1 )*( j + 1 ) - 1 ] = in[ row_start + j ];
		}
	}

	/* bubble sort */
	bool swap_happened = false;
	do
	{
		for ( int i = 1; i < filtersize*filtersize; ++i )
			if (neighbors[i] > neighbors[i-1]){
				int tmp = neighbors[ i ];
				neighbors[ i ] = neighbors[ i + 1 ];
				neighbors[ i + 1 ] = tmp;
			}
	} while ( swap_happened );

	/* set */
	out[ index ] = neighbors[ filtersize*filtersize / 2 ];
	free ( neighbors );
}

cudaError_t median_filter_gpu ( std::string, std::string, unsigned int );

int main()
{
	/* perform median filter with GPU */
    cudaError_t cudaStatus = median_filter_gpu("lena512.pgm","out.pgm", 7);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "median calculation failed!");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

/* wrap the kernel call here */
cudaError_t median_filter_gpu(std::string inputfilename, std::string outputfilename, unsigned int size)
{
	unsigned char * host_lena = NULL;
    unsigned char * dev_input = 0;
    unsigned char * dev_output = 0;
    cudaError_t cudaStatus;
	
	/* boilerplate malloc code as seen in the CUDA code */
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	/* load up lena, allocates memory if not given */
	unsigned int width;
	unsigned int height;
	sdkLoadPGM<unsigned char> ( inputfilename.c_str(), &host_lena, &width, &height );
	
	/* create space on card for lena IN */
	cudaStatus = cudaMalloc ( ( void** )&dev_input, im_size * im_size * sizeof ( unsigned char ) );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	/* create space on card for lena OUT */
	cudaStatus = cudaMalloc ( ( void** )&dev_output, im_size * im_size * sizeof ( unsigned char ) );
	if ( cudaStatus != cudaSuccess )
	{
		fprintf ( stderr, "cudaMalloc failed!" );
		goto Error;
	}

	/* copy host lena into card space */
	cudaStatus = cudaMemcpy ( dev_input, host_lena, im_size * im_size * sizeof ( unsigned char ), cudaMemcpyHostToDevice );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	/* define kernel parameters */
	dim3 threadsPerBlock ( 16 );
	dim3 numBlocks ( im_size / threadsPerBlock.x, im_size / threadsPerBlock.y );

    /* Launch a kernel on the GPU with 32 threads for each block */
    sobel_kernel<<<numBlocks, threadsPerBlock>>>(dev_input, dev_output);

	/* check what went wrong */
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "median_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
	/* finish up */
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching median_kernel!\n", cudaStatus);
        goto Error;
    }

	/* copy the data off */
	memset ( host_lena, 0, im_size*im_size );
	cudaStatus = cudaMemcpy ( host_lena, dev_output, im_size*im_size * sizeof ( unsigned char ), cudaMemcpyDeviceToHost );
	if ( cudaStatus != cudaSuccess )
	{
		fprintf ( stderr, "cudaMemcpy failed!" );
		goto Error;
	}

	sdkSavePGM ( outputfilename.c_str (), host_lena, width, height );

Error:
	free ( host_lena ); host_lena = NULL;
    cudaFree(dev_input);
    cudaFree(dev_output);
    
    return cudaStatus;
}
