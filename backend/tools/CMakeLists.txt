find_package(Threads REQUIRED)

# The server and its direct dependencies.
add_subdirectory(mtmd)
add_subdirectory("${PROJECT_SOURCE_DIR}/../frontend" "${CMAKE_BINARY_DIR}/frontend")
add_subdirectory(server)
