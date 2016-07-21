# scan source files
aux_source_directory(. DIR_LIB_SRCS)

# generate static library
add_library(poisson_solver ${DIR_LIB_SRCS})