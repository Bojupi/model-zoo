cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(ptree)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

find_package(Eigen3 CONFIG REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(_ptree_cpp ptree.cpp mve.cpp bindings.cpp)
target_link_libraries(_ptree_cpp PRIVATE Eigen3::Eigen pybind11::pybind11 OpenMP::OpenMP_CXX)
set_target_properties(_ptree_cpp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})