#include "common.cuh"
#include "sobel.cuh"
#include "cpu_functions.cu"
/* prototype for call below that wraps launching the sobel filter kernel */
inline void sobel_filter ( const std::string in, const std::string out);

int main (int argc, char* argv[])
{
	if ( argc != 3 )
	{
		std::cout << "Incorrect usage, execute with parameters: <input 512x512 .pgm image path> <output path>" << std::endl;
		return 1;
	}

	std::string input_file_path = std::string ( argv[ 1 ] );
	std::string outpu_file_path = std::string ( argv[ 2 ] );
	
	/* perform sobel filter with GPU */
	sobel_filter ( input_file_path, outpu_file_path );

    return 0;
}

/* wrap the kernel call here */
/*and there is a string map that correlates an input to the correct template function to execute */
void sobel_filter ( std::string inputfilename, std::string outputfilename )
{
	unsigned char * host_lena = NULL;
	unsigned char * dev_input = 0;
	unsigned char * dev_output = 0;

	cudaSetDevice ( 0 );

	/* load up lena, allocates memory if not given */
	unsigned int width;
	unsigned int height;
	sdkLoadPGM<unsigned char> ( inputfilename.c_str (), &host_lena, &width, &height );

	start ( "gpu timer" );

	/* create space on card for lena IN */
	cudaMalloc ( ( void** )&dev_input, IMAGE_SIZE * IMAGE_SIZE * sizeof ( unsigned char ) );

	/* create space on card for lena OUT */
	cudaMalloc ( ( void** )&dev_output, IMAGE_SIZE * IMAGE_SIZE * sizeof ( unsigned char ) );

	/* copy host lena into card space */
	cudaMemcpy ( dev_input, host_lena, IMAGE_SIZE * IMAGE_SIZE * sizeof ( unsigned char ), cudaMemcpyHostToDevice );

	/* define kernel parameters */
	dim3 threadsPerBlock ( 32 );
	dim3 numBlocks ( IMAGE_SIZE / threadsPerBlock.x, IMAGE_SIZE / threadsPerBlock.y );

	/* Launch a kernel on the GPU with 32 threads for each block */
	sobel_kernel <<<numBlocks, threadsPerBlock >>>( dev_input, dev_output );

	/* finish up */
	cudaError_t error = cudaDeviceSynchronize ();

	error = cudaGetLastError ();

	/* copy the data off */
	unsigned char out[ IMAGE_SIZE*IMAGE_SIZE ] = { 0 };
	cudaMemcpy ( out, dev_output, IMAGE_SIZE*IMAGE_SIZE * sizeof ( unsigned char ), cudaMemcpyDeviceToHost );

	auto time = get_time ( "gpu timer" );
	std::cout << "Time: " << time << std::endl;
/*
	float accuracy = calculate_accuracy ( &out[0], host_lena, size );
	std::cout << "Accuracy: " << accuracy*100 << "% of pixels were correct" << std::endl;*/
	/* save output file */
	sdkSavePGM ( outputfilename.c_str (), out, width, height );

	/* cleanup */
	free ( host_lena ); host_lena = NULL;
	cudaFree ( dev_input );
	cudaFree ( dev_output );

	cudaDeviceReset ();
}
