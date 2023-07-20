#include "python_submodule.h"

// See Pybind11 FAQ «How can I reduce the build time ?» :
// https://pybind11.readthedocs.io/en/stable/faq.html#how-can-i-reduce-the-build-time

//Python Binding and Time_Quad class instances.
PYBIND11_MODULE(autocorrelation, m)
{
    m.doc() = "Custum autocorrelation using OpenMP";
	init_autocorrelation(m);
}

