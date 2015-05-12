#include "common.cuh"
#include "median.cuh"
#include "median.h"

/* prototype for call below that wraps launching the median filter kernel */
cudaError_t median_filter_gpu ( std::string in, std::string out, std::string size);

int main()
{

	std::string in_file = "lena.pgm", out_file = "out.pgm", size = "3";
	/* perform median filter with GPU */
    cudaError_t cudaStatus = median_filter_gpu(in_file, out_file, size);
    
	/* clear the device */
	cudaStatus = cudaDeviceReset();

    return 0;
}

/* wrap the kernel call here */
cudaError_t median_filter_gpu(std::string inputfilename, std::string outputfilename, std::string size)
{
	unsigned char * host_lena = NULL;
    unsigned char * dev_input = 0;
    unsigned char * dev_output = 0;
    cudaError_t cudaStatus;
	
    cudaStatus = cudaSetDevice(0);

	/* load up lena, allocates memory if not given */
	unsigned int width;
	unsigned int height;
	sdkLoadPGM<unsigned char> ( inputfilename.c_str(), &host_lena, &width, &height );
	
	/* create space on card for lena IN */
	cudaStatus = cudaMalloc ( ( void** )&dev_input, IMAGE_SIZE * IMAGE_SIZE * sizeof ( unsigned char ) );

	/* create space on card for lena OUT */
	cudaStatus = cudaMalloc ( ( void** )&dev_output, IMAGE_SIZE * IMAGE_SIZE * sizeof ( unsigned char ) );

	/* copy host lena into card space */
	cudaStatus = cudaMemcpy ( dev_input, host_lena, IMAGE_SIZE * IMAGE_SIZE * sizeof ( unsigned char ), cudaMemcpyHostToDevice );

	/* define kernel parameters */
	dim3 threadsPerBlock ( 16 );
	dim3 numBlocks ( IMAGE_SIZE / threadsPerBlock.x, IMAGE_SIZE / threadsPerBlock.y );

    /* Launch a kernel on the GPU with 32 threads for each block */
    get_median_kernel(size) <<<numBlocks, threadsPerBlock>>>(dev_input, dev_output);

	/* check what went wrong */
    cudaStatus = cudaGetLastError();
    
	/* finish up */
    cudaStatus = cudaDeviceSynchronize();

	/* copy the data off */
	memset ( host_lena, 0, IMAGE_SIZE*IMAGE_SIZE );
	//cudaStatus = cudaMemcpy ( host_lena, dev_output, IMAGE_SIZE*IMAGE_SIZE * sizeof ( unsigned char ), cudaMemcpyDeviceToHost );

	sdkSavePGM ( outputfilename.c_str (), host_lena, width, height );

	/* cleanup */
	free ( host_lena ); host_lena = NULL;
    cudaFree(dev_input);
    cudaFree(dev_output);
    
    return cudaStatus;
}
