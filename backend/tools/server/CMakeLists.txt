include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by dgs-server and CLI

set(vTarget server-context)

add_library(${vTarget} STATIC
  server-chat.cpp
  server-chat.h
  server-task.cpp
  server-task.h
  server-queue.cpp
  server-queue.h
  server-common.cpp
  server-common.h
  server-decision.cpp
  server-decision.h
  server-context.cpp
  server-context.h
  server-stream.cpp
  server-stream.h
  server-tools.cpp
  server-tools.h
  server-mcp.cpp
  server-mcp.h
  server-monitor.cpp
  server-monitor.h
  server-schema.cpp
  server-schema.h
)

if (BUILD_SHARED_LIBS)
  set_target_properties(${vTarget} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${vTarget} PRIVATE ../mtmd)
target_include_directories(${vTarget} PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(${vTarget} PUBLIC dgs-common mtmd ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

# dgs-server-impl: server logic, reusable by app

set(vTarget dgs-server-impl)

file(READ "${PROJECT_SOURCE_DIR}/vendor/swagger-ui/swagger-ui-bundle.js" vSwaggerScript)
file(READ "${PROJECT_SOURCE_DIR}/vendor/swagger-ui/swagger-ui.css" vSwaggerStyle)
file(READ "${PROJECT_SOURCE_DIR}/../frontend/api.html" vSwaggerPage)
configure_file(server-doc-assets.h.in server-doc-assets.h @ONLY)

add_library(${vTarget}
  server-api-doc.cpp
  server-api-doc.h
  server.cpp
  server-shell-mcp.cpp
  server-shell-mcp.h
  server-ui-settings.cpp
  server-ui-settings.h
  server-http.cpp
  server-http.h
  server-models.cpp
  server-models.h
)
set_target_properties(${vTarget} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${vTarget} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${vTarget} PRIVATE ../mtmd ${PROJECT_SOURCE_DIR})
target_link_libraries(${vTarget} PUBLIC server-context dgs-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

add_dependencies(${vTarget} dgs-ui-assets)

if(DGS_TOOLS_INSTALL)
  install(TARGETS ${vTarget} LIBRARY)
endif()

# dgs-server executable

set(vTarget dgs-server)

add_executable(${vTarget} main.cpp)
install(TARGETS ${vTarget} RUNTIME)

target_link_libraries(${vTarget} PRIVATE dgs-server-impl)
target_compile_features(${vTarget} PRIVATE cxx_std_17)
