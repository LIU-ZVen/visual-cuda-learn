cmake_minimum_required(VERSION 3.26)

project(vcl_C LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

set(PYTHON_SUPPOTED_VERSIONS 3.10 3.11 3.12)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# set rpath
get_filename_component(TORCH_LIBRARIES_DIR "${TORCH_LIBRARIES}" DIRECTORY)
set(RPATH_VALUE $ORIGIN)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${RPATH_VALUE}/lib:${TORCH_LIBRARIES_DIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# build targets
file(GLOB SRC
    csrc/*.cpp
    csrc/**/*.cpp
    csrc/*.cu
    csrc/**/*.cu
)

pybind11_add_module(${PROJECT_NAME} ${SRC})

target_compile_definitions(${PROJECT_NAME} PRIVATE
    "-DTORCH_LIBRARY_NAME=${PROJECT_NAME}"
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES native)

target_include_directories(${PROJECT_NAME} PRIVATE
    csrc/include
    ${PYTHON_INCLUDE_PATH}
    ${pybind11_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
)

# Always install the extension library directly into the source tree's vcl/
install(
    TARGETS ${PROJECT_NAME}
    CONFIGURATIONS Debug Release
    LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/vcl"
)
