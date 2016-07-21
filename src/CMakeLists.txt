IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # Mac OS X specific code
    SET(OperatingSystem "Mac OS X")
    # add include path for Eigen3 (from homebrew)
    include_directories(/usr/local/include/eigen3)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    # Linux specific code
    SET(OperatingSystem "Linux")
    include_directories(/usr/include/eigen3)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

# scan source files
aux_source_directory(. DIR_LIB_SRCS)

# generate static library
add_library(poisson_solver ${DIR_LIB_SRCS})
