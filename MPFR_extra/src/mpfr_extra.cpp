#include <mpfr_extra.h>

using namespace std;

// For setting desired mpreal precision beforehand
void set_mpreal_precision(int d){
    // Before any mreal are created
    const int digits = d; // Setting high precision
    mpreal::set_default_prec(mpfr::digits2bits(digits));
}