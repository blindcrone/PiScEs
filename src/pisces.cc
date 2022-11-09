#include "pisces.h"
#include <pybind11/functional.h>
PYBIND11_MODULE(pisces, m) {
    m.def("PiScEs", &PiScEs);
}
