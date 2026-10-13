cmake_minimum_required(VERSION 3.19)
project(dense-gguf-server LANGUAGES C CXX)

# Build the engine, the server and the web interface from the backend tree.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../backend" "${CMAKE_CURRENT_BINARY_DIR}/backend")
