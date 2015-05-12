#ifndef BDS_SCM_CPU
#define BDS_SCM_CPU "5/11/2015 Ben Sammons"

#include "common.h"

/*CPU implementation of median kernel*/



template<int FILTER_WIDTH>
struct mediator
{
	
	static inline std::vector<int> starters ()
	{
		std::vector<int> start_values;
		for ( int i = 0; i < FILTER_WIDTH*FILTER_WIDTH; ++i )
			start_values.push_back ( ( i % 2 == 0 ) ? 255 : 0 );
		return start_values;
	}

	template<int num>
	static inline void get_neighbors ( const unsigned char *in, std::vector<int>& neighbors, const int top_left_index )
	{
		neighbors[ num ] = in[ top_left_index + ( num % FILTER_WIDTH ) + ( num / FILTER_WIDTH ) * IMAGE_SIZE ];
		get_neighbors<num - 1> ( in, neighbors, top_left_index );
	}
	template<>
	static inline void get_neighbors<-1> ( const unsigned char *in, std::vector<int>& neighbors, const int top_left_index ) {}

	template<int i>
	static inline void get_medians_i ( const unsigned char*& in, unsigned char*& out )
	{
		get_medians_j<IMAGE_SIZE - ( FILTER_WIDTH >> 1 )> ( in, out, i );
		get_medians_i<i - IMAGE_SIZE> ( in, out );
	}

	template<>
	static inline void get_medians_i<IMAGE_SIZE*( (FILTER_WIDTH >> 1)-1 )> ( const unsigned char*& in, unsigned char*& out ) {}

	template<int j>
	static inline void get_medians_j ( const unsigned char*& in, unsigned char*& out, const int& i )
	{
		const int FILTER_WIDTH_SQUARED = FILTER_WIDTH*FILTER_WIDTH;
		
		const int column = j;
		const int index = i + j;
		auto neighbors = starters();
		mediator<FILTER_WIDTH>::get_neighbors<FILTER_WIDTH_SQUARED - 1> ( in, neighbors, index - ( FILTER_WIDTH >> 1 ) - IMAGE_SIZE*( FILTER_WIDTH >> 1 ) );
		std::sort ( &neighbors[ 0 ], &neighbors[ FILTER_WIDTH_SQUARED - 1 ] );
		out[index] = neighbors[ FILTER_WIDTH_SQUARED / 2 ];
		get_medians_j<j - 1> ( in, out, i );
	}

	template<>
	static inline void get_medians_j<(FILTER_WIDTH >> 1)-1> ( const unsigned char*& in, unsigned char*& out, const int& i ) {}
};

//template<int FILTER_WIDTH>
//struct mediator < 0 >
//{
//
//	static inline void get_neighbors ( const unsigned char *in, vector<int>& neighbors )
//};


template<int FILTER_WIDTH>
void median_kernel_cpu ( const unsigned char *in, unsigned char *out )
{
	/* assume image is IMAGE_WIDTH by IMAGE_WIDTH */
	/* do the easy part first */
	#pragma omp parallel for
	mediator<FILTER_WIDTH>::get_medians_i<IMAGE_SIZE*IMAGE_SIZE - (( FILTER_WIDTH >> 1 ) * IMAGE_SIZE)> ( in, out );
}

#endif