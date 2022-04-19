cmake_minimum_required(VERSION 3.1 FATAL_ERROR)


# general configuration for compilation:
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -g")

enable_language(C CXX)




# if(MPI)
#     set(MPI_ASSUME_NO_BUILTIN_MPI True)
#     find_package(MPI REQUIRED COMPONENTS C CXX)
#     include_directories(MPI_C_INCLUDE_DIRECTORIES)
#     include_directories(MPI_CXX_INCLUDE_DIRECTORIES)
#     add_definitions(-DLARCV_MPI)
# endif()


project(mlqm)

set(CMAKE_PACKAGE_DIR "mlqm")



include_directories("./")

# IO needs HDF5:
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
message("TORCH FOUND: ${TORCH_FOUND}")
message("TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message("TORCH Libraries: ${TORCH_LIBRARIES}")
message("TORCH FLAGS: ${TORCH_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# # From the torch example:
# add_executable(dcgan dcgan.cpp)
# target_link_libraries(dcgan "${TORCH_LIBRARIES}")
# set_property(TARGET dcgan PROPERTY CXX_STANDARD 14)



# if(OPENMP)
#   find_package(OpenMP)
#   if(OpenMP_CXX_FOUND)
#       MESSAGE(STATUS "OpenMP Include directories ${OpenMP_CXX_INCLUDE_DIRS}")
#       include_directories(${OpenMP_C_INCLUDE_DIRS})
#       include_directories(${OpenMP_CXX_INCLUDE_DIRS})
#       add_definitions(-DLARCV_OPENMP)
#   endif()
# endif()

# This package needs numpy::

# # Many packages need python:
# find_package(PythonLibs REQUIRED)
# include_directories(${PYTHON_INCLUDE_DIRS})



# # This will come back for configuration:

# # # Add the json library
# # set(JSON_BuildTests OFF CACHE INTERNAL "")
# # add_subdirectory(json)
# # message("NLOHMANN_JSON_INCLUDE_INSTALL_DIR: ${NLOHMANN_JSON_INCLUDE_INSTALL_DIR}")
# # include_directories(json/include)


# message("Adding pybind11")
# find_package(pybind11 REQUIRED)
# include_directories(${PYBIND11_INCLUDE_DIR})



# # GIves bindings between json and python
# add_subdirectory(pybind11_json)
# include_directories(pybind11_json/include)




add_subdirectory(mlqm)
