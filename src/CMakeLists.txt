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

#Main dependency is torch:
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
message("TORCH FOUND: ${TORCH_FOUND}")
message("TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message("TORCH Libraries: ${TORCH_LIBRARIES}")
message("TORCH FLAGS: ${TORCH_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# Try for OpenMP for task parallelization:
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    MESSAGE(STATUS "OpenMP CXX Include directories ${OpenMP_CXX_INCLUDE_DIRS}")
    MESSAGE(STATUS "OpenMP CXX LibNames ${OpenMP_CXX_LIB_NAMES}")
    MESSAGE(STATUS "OpenMP CXX Library ${OpenMP_CXX_LIBRARY}")
    MESSAGE(STATUS "OpenMP CXX Include directories ${OpenMP_CXX_INCLUDE_DIRS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
    add_definitions(-DOPENMP_FOUND)
endif()

# Add the external folders for all the extras
# (logging, tensorboard, tests, json)

# Add the json library
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory("json")
include_directories("json/include")

# Add Toml Library:

# Here is the testing library:
add_subdirectory("Catch2")

# Add the logging directory:
add_subdirectory("plog")
include_directories("plog/include")

# # Add tensorboard:
# add_subdirectory("tensorboard_logger")
# include_directories("tensorboard_logger/include")


add_subdirectory(mlqm)
