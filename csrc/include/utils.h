#ifndef CSRC_INCLUDE_UTILS_H_
#define CSRC_INCLUDE_UTILS_H_

#include <Python.h>

#define STRINGIFY(x) #x
#define TO_STRING(x) STRINGIFY(x)

#define CONCATENATE(x, y) x##y

#define TORCH_LIBRARY_WRAPPER(name, module) TORCH_LIBRARY(name, module)

#define REGISTER_EXTENSION(name)                                               \
  PyMODINIT_FUNC CONCATENATE(PyInit_, name)(void) {                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        TO_STRING(name), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }

#endif  // CSRC_INCLUDE_UTILS_H_
