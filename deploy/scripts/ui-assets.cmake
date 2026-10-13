# Included from frontend/CMakeLists.txt; paths are relative to the frontend source directory.
find_package(Python3 3.13 REQUIRED COMPONENTS Interpreter)
set(vUiCpp "${CMAKE_CURRENT_BINARY_DIR}/ui.cpp")
set(vUiHeader "${CMAKE_CURRENT_BINARY_DIR}/ui.h")
set(vUiDist "${CMAKE_CURRENT_SOURCE_DIR}/dist")
option(DGS_USE_LOCAL_UI "Embed an already built frontend/dist without running npm" OFF)

if(DGS_BUILD_UI AND DGS_USE_LOCAL_UI)
  if(NOT EXISTS "${vUiDist}/index.html")
    message(FATAL_ERROR "DGS_USE_LOCAL_UI requires a complete frontend/dist built from these sources")
  endif()
  file(GLOB_RECURSE aUiDependencies CONFIGURE_DEPENDS "${vUiDist}/*")
elseif(DGS_BUILD_UI)
  find_program(vNpmExecutable npm REQUIRED)
  file(GLOB_RECURSE aUiSources CONFIGURE_DEPENDS ${aUiSourceGlobs})
  add_custom_command(OUTPUT "${vUiDist}/index.html"
    COMMAND "${vNpmExecutable}" ci --no-audit --no-fund
    COMMAND "${CMAKE_COMMAND}" -E env "DGS_BUILD_NUMBER=${DGS_BUILD_NUMBER}" "${vNpmExecutable}" run build
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS ${aUiSources} ${aUiSourceFiles}
    COMMENT "Building the DGS web application" VERBATIM)
  set(aUiDependencies "${vUiDist}/index.html")
endif()
add_custom_command(OUTPUT "${vUiCpp}" "${vUiHeader}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/embed.py"
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}" "${DGS_BUILD_UI}"
  DEPENDS ${aUiDependencies} ui.cpp.in ui.h.in "${CMAKE_CURRENT_SOURCE_DIR}/embed.py"
  COMMENT "Embedding DGS web assets" VERBATIM)
