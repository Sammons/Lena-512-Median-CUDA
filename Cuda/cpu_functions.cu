#ifndef BDS_CPU
#define BDS_CPU "5/11/2015 Ben Sammons"

#include "common.cuh"

/* accuracy check vs some median filter*/
template<int filtersize>
float calculate_accuracy ( unsigned char*& gpu_computed, unsigned char*& original )
{
	int total_pixel_count = IMAGE_SIZE * IMAGE_SIZE;
	std::vector<int> default_neighbors ( filtersize*filtersize );
	for ( int i = 0; i < filtersize*filtersize; ++i )
	{
		default_neighbors[ i ] = ( ( ( i % 2 ) == 0 ) ? 255 : 0 );
	}
	/* iterate over every pixel */
	int correct = 0;
	for ( int i = 0; i < total_pixel_count; ++i )
	{
		/* get top left */
		int top_left = ( i - ( filtersize /2 )*IMAGE_SIZE - ( filtersize / 2 ) );
		std::vector<int> neighbors = default_neighbors;
		for ( int k = 0; k < filtersize; k++ )
		{
			const int row_start = top_left + IMAGE_SIZE*k;
			for ( int j = 0; j < filtersize; j++ )
			{
				if ( in_bounds_cpu ( row_start + j ) )
					neighbors[ ( k + 1 )*( j + 1 ) - 1 ] = original[ row_start + j ];
			}
		}
		std::sort ( neighbors.begin (), neighbors.end () );
		const int expected = neighbors[ ( filtersize*filtersize ) / 2 ];
		const int provided = gpu_computed[ i ];
		if ( abs(expected - provided) == 0 )
			++correct;
	}

	return static_cast< float >( correct ) / static_cast< float >( IMAGE_SIZE*IMAGE_SIZE );
}

template float calculate_accuracy < 3 > ( unsigned char*& gpu_computed, unsigned char*& original );
template float calculate_accuracy < 5 > ( unsigned char*& gpu_computed, unsigned char*& original );
template float calculate_accuracy < 7 > ( unsigned char*& gpu_computed, unsigned char*& original );
template float calculate_accuracy < 9 > ( unsigned char*& gpu_computed, unsigned char*& original );
template float calculate_accuracy < 11 > ( unsigned char*& gpu_computed, unsigned char*& original );
template float calculate_accuracy < 13 > ( unsigned char*& gpu_computed, unsigned char*& original );
template float calculate_accuracy < 15 > ( unsigned char*& gpu_computed, unsigned char*& original );

float calculate_accuracy ( unsigned char* gpu_computed, unsigned char* original, std::string size )
{
	if ( size == "3" ) return calculate_accuracy<3> ( gpu_computed, original );
	if ( size == "5" ) return calculate_accuracy<5> ( gpu_computed, original );
	if ( size == "7" ) return calculate_accuracy<7> ( gpu_computed, original );
	if ( size == "9" ) return calculate_accuracy<9> ( gpu_computed, original );
	if ( size == "11" ) return calculate_accuracy<11> ( gpu_computed, original );
	if ( size == "13" ) return calculate_accuracy<13> ( gpu_computed, original );
	if ( size == "15" ) return calculate_accuracy<15> ( gpu_computed, original );
	std::cout << "no instance of grading function capable of grading the given size" << std::endl;
	return 0.0;
}



#endif