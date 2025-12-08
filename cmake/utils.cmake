# This cmake function is adapted from vllm /Users/ganyi/workspace/vllm-ascend/cmake/utils.cmake
# Define a target named `GPU_MOD_NAME` for a single extension. The
# arguments are:
#
# DESTINATION <dest>         - Module destination directory.
# LANGUAGE <lang>            - The GPU language for this module, e.g CUDA, HIP,
#                              etc.
# SOURCES <sources>          - List of source files relative to CMakeLists.txt
#                              directory.
#
# Optional arguments:
#
# ARCHITECTURES <arches>     - A list of target GPU architectures in cmake
#                              format.
#                              Refer `CMAKE_CUDA_ARCHITECTURES` documentation
#                              and `CMAKE_HIP_ARCHITECTURES` for more info.
#                              ARCHITECTURES will use cmake's defaults if
#                              not provided.
# COMPILE_FLAGS <flags>      - Extra compiler flags passed to NVCC/hip.
# INCLUDE_DIRECTORIES <dirs> - Extra include directories.
# LIBRARIES <libraries>      - Extra link libraries.
# WITH_SOABI                 - Generate library with python SOABI suffix name.
# USE_SABI <version>         - Use python stable api <version>
#
# Note: optimization level/debug info is set via cmake build type.
#
function(define_vcl_extension_target GPU_MOD_NAME)
  cmake_parse_arguments(PARSE_ARGV 1
    GPU
    "WITH_SOABI"  # 布尔型参数
    "DESTINATION;USE_SABI"  # 单值参数
    "SOURCES;INCLUDE_DIRECTORIES;LINK_DIRECTORIES;LIBRARIES;COMPILE_FLAGS"  # 列表参数
  )

  # --------------------------
  # 1. checks
  # --------------------------
  if(NOT GPU_LANGUAGE)
    set(GPU_LANGUAGE "CXX")
  endif()

  if(NOT GPU_DESTINATION)
    set(GPU_DESTINATION ".")
  endif()

  if(NOT GPU_SOURCES)
    message(FATAL_ERROR "define_vcl_extension_target(${GPU_MOD_NAME}) missing SOURCES argument.")
  endif()

  # --------------------------
  # 2. generate python module
  # --------------------------
  # Python_add_library(${GPU_MOD_NAME} MODULE)
  pybind11_add_module(${GPU_MOD_NAME} ${GPU_SOURCES})

  # --------------------------
  # 3. macros
  # --------------------------
  target_compile_definitions(${GPU_MOD_NAME} PRIVATE
    "-DTORCH_EXTENSION_NAME=${GPU_MOD_NAME}"
  )

  set_property(TARGET ${GPU_MOD_NAME} PROPERTY CXX_STANDARD 17)

  if (GPU_COMPILE_FLAGS)
    target_compile_options(${GPU_MOD_NAME} PRIVATE
      $<$<COMPILE_LANGUAGE:${GPU_LANGUAGE}>:${GPU_COMPILE_FLAGS}>
    )
  endif()

  # --------------------------
  # 4. include directories
  # --------------------------
  target_include_directories(${GPU_MOD_NAME} PRIVATE
    ${pybind11_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_PATH}
    ${TORCH_INCLUDE_DIRS}
    ${GPU_INCLUDE_DIRECTORIES}
  )

  # --------------------------
  # 5. link libraries
  # --------------------------
  target_link_directories(${GPU_MOD_NAME} PRIVATE
    ${GPU_LINK_DIRECTORIES}
  )

  target_link_libraries(${GPU_MOD_NAME} PRIVATE
    ${TORCH_LIBRARIES}
    ${GPU_LIBRARIES}
  )

  # --------------------------
  # 6. install
  # --------------------------
  install(TARGETS ${GPU_MOD_NAME} LIBRARY DESTINATION ${GPU_DESTINATION})
  message(STATUS "Defined ADA extension target: ${GPU_MOD_NAME}")
endfunction()
