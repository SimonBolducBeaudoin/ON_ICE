#pragma once

#ifdef _WIN32_WINNT
    #include "mpreal.h"
#else
    #include <mpreal.h>
#endif

using namespace std;
using mpfr::mpreal;

// For setting desired mpreal precision beforehand
void set_mpreal_precision(int d);
