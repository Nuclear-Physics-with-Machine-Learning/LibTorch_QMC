set(name optimizers)


# Get all the source files:
file(GLOB SOURCES *.cxx)
file(GLOB HEADERS *.h)

# Add a shared library
add_library(${name} OBJECT ${SOURCES})

