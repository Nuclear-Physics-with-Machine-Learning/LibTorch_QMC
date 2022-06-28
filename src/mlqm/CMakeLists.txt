add_subdirectory(config)
add_subdirectory(models)
add_subdirectory(sampler)
add_subdirectory(hamiltonians)
add_subdirectory(optimizers)


# This is admittedly pretty clunky, but it gets the job done to make one object file:

add_library(libmlqm SHARED
    $<TARGET_OBJECTS:config>
    $<TARGET_OBJECTS:models>
    $<TARGET_OBJECTS:sampler>
    $<TARGET_OBJECTS:hamiltonians>
    $<TARGET_OBJECTS:optimizers>
)


if(UNIX AND NOT APPLE)
    set_target_properties(libmlqm PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib/"
        IMPORTED_LOCATION "$ORIGIN/../lib/")
elseif(APPLE)
    set_target_properties(libmlqm PROPERTIES
        INSTALL_RPATH "@loader_path/../lib/"
        IMPORTED_LOCATION "@loader_path/../lib/")
endif()

# if (OpenMP_CXX_FOUND)
#     message("Linking against openmp")
#     message("OpenMP_CXX_LIBRARY: " ${OpenMP_CXX_LIBRARIES})
#     message("OpenMP_C_LIBRARY: " ${OpenMP_C_LIBRARIES})
#     target_link_libraries(mlqm OpenMP::OpenMP_C)
# endif()

# if (MPI_FOUND)
#     message("Linking against MPI")
#     message("OpenMP_CXX_LIBRARY: " ${MPI_CXX_LIBRARIES})
#     message("OpenMP_C_LIBRARY: " ${MPI_C_LIBRARIES})
#     target_link_libraries(mlqm ${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES})
# endif()


# Link against python:
# target_link_libraries(mlqm ${PYTHON_LIBRARIES})

# Link against pytorch:
target_link_libraries(libmlqm ${TORCH_LIBRARIES})



###############################

# Link against the json library:
target_link_libraries(libmlqm nlohmann_json::nlohmann_json)



# pybind11_add_module(pymlqm SHARED mlqm.cxx)

# # Add a shared library
# set_target_properties(pymlqm PROPERTIES SKIP_BUILD_RPATH FALSE)

# # If building with setuptools, CMake will not be performing the install
# set_target_properties(pymlqm PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

# target_link_libraries(pymlqm PRIVATE mlqm)

# if(UNIX AND NOT APPLE)
#     set_target_properties(pymlqm PROPERTIES INSTALL_RPATH "$ORIGIN/lib/")
# elseif(APPLE)
#     set_target_properties(pymlqm PROPERTIES INSTALL_RPATH "@loader_path/lib/")
# endif()

# set_target_properties(pymlqm PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)


###############################

# if (MPI)
#     target_link_libraries(mlqm ${MPI_LIBRARIES})
# endif()

add_executable(mlqm mlqm.cxx)

# Link the mlqm executable against
target_link_libraries(mlqm libmlqm)


# Install mlqm shared library
install(TARGETS libmlqm
    LIBRARY DESTINATION ${CMAKE_PACKAGE_DIR}/lib
    ARCHIVE DESTINATION ${CMAKE_PACKAGE_DIR}/lib
    RUNTIME DESTINATION ${CMAKE_PACKAGE_DIR}/bin
    INCLUDES DESTINATION ${CMAKE_PACKAGE_DIR}/include
    )

# Install mlqm executable
install(TARGETS mlqm
    LIBRARY DESTINATION ${CMAKE_PACKAGE_DIR}/lib
    ARCHIVE DESTINATION ${CMAKE_PACKAGE_DIR}/lib
    RUNTIME DESTINATION ${CMAKE_PACKAGE_DIR}/bin
    INCLUDES DESTINATION ${CMAKE_PACKAGE_DIR}/include
    )

# Add a test executable:
add_executable(tests tests.cxx)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain libmlqm)

# Add a benchmark executable:
add_executable(benchmark benchmark_jacobian.cxx)
target_link_libraries(benchmark PRIVATE Catch2::Catch2WithMain libmlqm)


# # This is from the docs ...
# # These tests need their own main
# add_executable(custom-main-tests test.cpp test-main.cpp)
# target_link_libraries(custom-main-tests PRIVATE Catch2::Catch2)


# # Install mlqm wrapper library
# install(TARGETS pymlqm
#     LIBRARY DESTINATION ${CMAKE_PACKAGE_DIR}
#     ARCHIVE DESTINATION ${CMAKE_PACKAGE_DIR}
#     # RUNTIME DESTINATION ${CMAKE_PACKAGE_DIR}/bin
#     # INCLUDES DESTINATION ${CMAKE_PACKAGE_DIR}/include
#     )
