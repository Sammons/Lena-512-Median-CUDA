#ifndef BDS_COMMON_CPU
#define BDS_COMMON_CPU "stuff everybody wants, but probably doesn't need"

/* things that you configure for compilation */
#define IMAGE_SIZE 512

/* compiled things that help */

#include <string>
#include <iostream>
#include <map>
#include <vector>

/* stupid simple helper functions */
inline const bool const in_bounds_cpu ( const int x ) { return ( x >= 0 && x < IMAGE_SIZE*IMAGE_SIZE ); }

#include "inc/helper_image.h"

#endif