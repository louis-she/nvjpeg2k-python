cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(nvjpeg2k_python LANGUAGES CXX CUDA)

if (NOT NVJPEG2K_PATH)
  message(FATAL_ERROR, "NVJPEG2K_PATH not set")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(FILESYS -lstdc++fs)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

include_directories(
  ${NVJPEG2K_PATH}/include
  SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

add_subdirectory(extern/pybind11)

find_library(NVJPEG2K_LIB NAMES nvjpeg2k_static PATHS ${NVJPEG2K_PATH}/lib)
find_library(CUDART_LIB NAMES cudart PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
# # find_package(PythonLibsNew REQUIRED)

add_library(nvjpeg2k SHARED "nvjpeg2k.cpp")
add_executable(test_nvjpeg2k "nvjpeg2k.cpp")

SET_TARGET_PROPERTIES(nvjpeg2k PROPERTIES PREFIX "")
target_link_libraries(nvjpeg2k PUBLIC ${NVJPEG2K_LIB} ${CUDART_LIB} ${FILESYS} pybind11::module)
target_link_libraries(test_nvjpeg2k PUBLIC ${NVJPEG2K_LIB} ${CUDART_LIB} ${FILESYS} pybind11::pybind11 pybind11::embed pybind11::module)
